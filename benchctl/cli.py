"""benchctl CLI — ARM 端侧推理基准测试统一入口."""
import click
import json
import datetime
import subprocess as _sp
import os
from pathlib import Path

from .db import Database, DB_PATH


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """ARM Inference Benchmark CLI — 统一基准测试工具."""
    pass


@cli.group()
def db():
    """数据库管理命令."""
    pass


@db.command("init")
def db_init():
    """初始化数据库."""
    d = Database()
    d.close()
    click.echo(f"✅ Database initialized: {DB_PATH}")


@db.command("stats")
def db_stats():
    """数据库统计."""
    d = Database()
    total = d.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    click.echo(f"Total runs: {total}")
    d.close()


@db.command("history")
@click.option("--framework", "-f", required=True, help="框架名 (mnn/ort/tvm/llamacpp)")
@click.option("--model", "-m", required=True, help="模型名")
@click.option("--limit", "-n", default=10, help="返回条数")
def db_history(framework, model, limit):
    """查询历史记录."""
    d = Database()
    rows = d.history(framework, model, limit)
    if not rows:
        click.echo("No records found.")
    else:
        for r in rows:
            click.echo(json.dumps(dict(r), ensure_ascii=False, indent=2))
    d.close()


@db.command("latest")
@click.option("--framework", "-f", required=True, help="框架名")
@click.option("--model", "-m", required=True, help="模型名")
def db_latest(framework, model):
    """获取最新一条记录."""
    d = Database()
    row = d.latest(framework, model)
    if row:
        click.echo(json.dumps(dict(row), ensure_ascii=False, indent=2))
    else:
        click.echo("No record found.")
    d.close()


@db.command("compare")
@click.option("--frameworks", "-f", required=True, help="框架列表，逗号分隔 (mnn,ort,tvm)")
@click.option("--model", "-m", required=True, help="模型名")
def db_compare(frameworks, model):
    """横向对比多个框架的最新数据."""
    fw_list = [fw.strip() for fw in frameworks.split(",")]
    d = Database()
    results = d.compare(fw_list, model)
    if not results:
        click.echo("No records found.")
    else:
        for r in results:
            click.echo(f"\n--- {r['framework']} ---")
            click.echo(json.dumps(dict(r), ensure_ascii=False, indent=2))
    d.close()


# ---------------------------------------------------------------------------
# run 命令
# ---------------------------------------------------------------------------
from .runner import AdbRunner
from .db import Database
from .tracks.base import TrackConfig
from .tracks.cnn import CNNTrack
from .tracks.llm import LLMTrack
from .tracks.single_op import SingleOpTrack

TRACKS = {
    "cnn": CNNTrack(),
    "llm": LLMTrack(),
    "single_op": SingleOpTrack(),
}


@cli.command()
@click.argument("track", type=click.Choice(["cnn", "llm", "single_op"]))
@click.argument("model")
@click.option(
    "-f", "--frameworks", default="mnn,ort",
    help="逗号分隔的框架列表 (mnn,ort,tvm,llamacpp)"
)
@click.option("-p", "--precision", default="fp32")
@click.option("-t", "--threads", default=4, type=int)
@click.option("-w", "--warmup", default=10, type=int)
@click.option("-r", "--runs", default=100, type=int)
@click.option("--no-save", is_flag=True, help="不保存到数据库")
def run(track, model, frameworks, precision, threads, warmup, runs, no_save):
    """执行基准测试.

    \b
    TRACK: cnn (CV/NLP模型) | llm (大语言模型) | single_op (单算子)
    MODEL: resnet50 | mobilenetv2 | qwen3-4b | matmul | ...
    """
    track_obj = TRACKS[track]
    fw_list = [f.strip() for f in frameworks.split(",")]

    config = TrackConfig(
        track=track,
        binary=track_obj.binary_name(),
        model=model,
        frameworks=fw_list,
        precision=precision,
        threads=threads,
        warmup=warmup,
        runs=runs,
    )

    runner = AdbRunner()

    # Step 1: 设备握手
    click.echo("设备握手...")
    try:
        device_info = runner.check_device()
    except RuntimeError as e:
        click.echo(f"设备连接失败: {e}", err=True)
        raise click.Abort()
    click.echo(f"  设备: {device_info['model']} / {device_info['soc']}")

    # Step 2: 推送二进制和库
    click.echo(f"推送 {track_obj.binary_name()}...")
    runner.push_binary(track_obj.binary_name())
    runner.push_libs()

    # Step 3: 获取设备温度
    temp = runner.get_device_temp()
    if temp:
        click.echo(f"设备温度: {temp}°C")

    # Step 4: 运行 benchmark
    args = track_obj.build_cli_args(config)
    click.echo(f"运行: {track_obj.binary_name()} {' '.join(args)}")
    results = runner.run_benchmark(track_obj.binary_name(), args)

    # Step 5: 保存结果
    if not no_save and results:
        db = Database()
        run_id = (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            + "-"
            + os.urandom(4).hex()
        )
        try:
            git_commit = (
                _sp.run(
                    ["git", "log", "--oneline", "-1"],
                    capture_output=True, text=True, cwd=runner.build_dir.parent
                )
                .stdout.strip()
            )
        except Exception:
            git_commit = "unknown"

        for r in results:
            r["run_id"] = run_id
            r["timestamp"] = datetime.datetime.now().isoformat()
            r["git_commit"] = git_commit
            r["track"] = track
            r["device_model"] = device_info["model"]
            r["device_temp"] = temp
            r["test_runs"] = runs
            # 确保必填字段存在
            for key in ("framework", "model", "precision", "threads", "warmup"):
                if key not in r:
                    r[key] = config.__dict__.get(key, "")
            db.insert(r)
        db.close()
        click.echo(f"已保存 {len(results)} 条结果到数据库")

    # Step 6: 打印摘要
    click.echo("\n结果摘要:")
    for r in results:
        m = r.get("metrics", {})
        click.echo(
            f"  {r.get('framework', '?'):12s}"
            f" | p50={m.get('p50_ms', 0):6.2f}ms"
            f" | p99={m.get('p99_ms', 0):6.2f}ms"
            f" | fps={m.get('throughput_fps', 0):6.1f}"
        )


# ---------------------------------------------------------------------------
# history 命令 — 纵向对比 + 回归检测
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("framework")
@click.argument("model")
@click.option("--last", default=10, type=int, help="显示最近 N 条")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def history(framework, model, last, json_output):
    """查看框架+模型的历史趋势 (纵向对比).

    \b
    FRAMEWORK: mnn | ort | tvm | llamacpp
    MODEL: resnet50 | mobilenetv2 | qwen3-4b
    """
    db = Database()
    rows = db.history(framework, model, limit=last)
    db.close()

    if json_output:
        import json as j
        click.echo(j.dumps(rows, indent=2))
        return

    if not rows:
        click.echo(f"没有 {framework}/{model} 的历史记录")
        return

    # 计算 baseline (最早的一条)
    baseline = rows[-1]
    b_metrics = json.loads(baseline["metrics_json"])
    b_p50 = b_metrics.get("p50_ms", 0)

    click.echo(f"\n{framework}/{model} 性能趋势 (最近 {last} 条)\n")
    click.echo(f"{'日期':12s} {'commit':10s} {'p50(ms)':>10s} {'变化':>10s} {'累计':>8s}")
    click.echo("-" * 56)

    for r in reversed(rows):  # 正序显示
        m = json.loads(r["metrics_json"])
        p50 = m.get("p50_ms", 0)
        delta = (p50 - b_p50) / b_p50 * 100 if b_p50 else 0
        cumulative = "BASE" if r == baseline else f"{delta:+.1f}%"

        click.echo(f"{r['timestamp'][:10]:12s} "
                   f"{r['git_commit'][:8]:10s} "
                   f"{p50:10.2f} "
                   f"{delta:+9.1f}% "
                   f"{cumulative:>8s}")

    # 回归检测
    if len(rows) >= 2:
        latest = json.loads(rows[0]["metrics_json"])
        prev = json.loads(rows[1]["metrics_json"])
        l_p50 = latest.get("p50_ms", 0)
        p_p50 = prev.get("p50_ms", 0)
        if p_p50 > 0 and l_p50 > p_p50 * 1.05:
            click.echo(f"\n回归警告: 最新 commit 比前一次慢 {(l_p50/p_p50 - 1)*100:.1f}%")
        elif p_p50 > 0:
            click.echo(f"\n无回归 (最新 vs 前次: {(l_p50/p_p50 - 1)*100:.1f}%)")


# ---------------------------------------------------------------------------
# verify 命令 — 原生工具验证层
# ---------------------------------------------------------------------------
from .verify import NativeVerifier


@cli.command()
@click.argument("framework")
@click.argument("model")
@click.option("-t", "--threads", default=4, type=int)
def verify(framework, model, threads):
    """验证 Harness 结果的可信度 (调用框架原生工具).

    \b
    FRAMEWORK: mnn | ort
    MODEL: resnet50 | mobilenetv2
    """
    db = Database()
    latest = db.latest(framework, model)
    db.close()

    if not latest:
        click.echo(f"没有 {framework}/{model} 的 Harness 数据, 请先执行 benchctl run")
        return

    harness_metrics = json.loads(latest["metrics_json"])
    harness_ms = harness_metrics.get("p50_ms", 0)

    click.echo(f"\n验证 {framework}/{model}:")
    click.echo(f"   Harness p50: {harness_ms:.2f} ms")

    verifier = NativeVerifier()
    if framework == "mnn":
        result = verifier.verify_mnn(model, threads, harness_ms)
    elif framework == "ort":
        result = verifier.verify_ort(model, threads, harness_ms)
    else:
        click.echo(f"   {framework} 暂不支持原生工具验证")
        return

    if result and "error" not in result:
        verdict_icon = "pass" if result["verdict"] == "trusted" else "fail"
        click.echo(f"   原生工具: {result['native_tool']}")
        click.echo(f"   原生结果: {result['native_result_ms']:.2f} ms")
        click.echo(f"   偏差:     {result['deviation_pct']:+.1f}%")
        click.echo(f"   可信度:   {verdict_icon} {result['verdict']}")
    else:
        click.echo(f"   验证失败: {result.get('error', 'unknown')}")


# ---------------------------------------------------------------------------
# export 命令 — JSON/HTML 报告导出
# ---------------------------------------------------------------------------
from .report import export_json, generate_html_compare


@cli.command()
@click.argument("model")
@click.option("-f", "--frameworks", default="mnn,ort,tvm",
              help="逗号分隔框架列表")
@click.option("-o", "--output", type=click.Path(), default=None,
              help="输出文件路径")
@click.option("--format", "fmt", type=click.Choice(["json", "html"]), default="html",
              help="输出格式")
def export(model, frameworks, output, fmt):
    """导出 benchmark 结果.

    \b
    MODEL: resnet50 | mobilenetv2 | ...
    """
    fw_list = [f.strip() for f in frameworks.split(",")]

    if fmt == "json":
        out = export_json(fw_list, model, Path(output) if output else None)
        click.echo(f"JSON 导出: {out}")
    elif fmt == "html":
        out = generate_html_compare(fw_list, model, Path(output) if output else None)
        click.echo(f"HTML 报告: {out}")


if __name__ == "__main__":
    cli()
