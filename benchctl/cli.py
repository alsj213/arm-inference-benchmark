"""benchctl CLI — ARM 端侧推理基准测试统一入口."""
import click
import json

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


if __name__ == "__main__":
    cli()
