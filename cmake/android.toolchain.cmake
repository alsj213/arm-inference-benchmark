# Android NDK toolchain file for CMake
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a \
#         -DANDROID_PLATFORM=android-29 \
#         ..

# If you have NDK installed in a custom path, set it here:
# set(ANDROID_NDK /path/to/your/android-ndk-rxxxxx)

if(NOT DEFINED ANDROID_NDK)
    if(DEFINED ENV{ANDROID_NDK})
        set(ANDROID_NDK $ENV{ANDROID_NDK})
    else()
        # Try common locations
        if(EXISTS /opt/android-ndk)
            set(ANDROID_NDK /opt/android-ndk)
        elseif(EXISTS $ENV{HOME}/android-ndk)
            set(ANDROID_NDK $ENV{HOME}/android-ndk)
        elseif(EXISTS /usr/local/android-ndk)
            set(ANDROID_NDK /usr/local/android-ndk)
        endif()
    endif()
endif()

# Target Android platform version (minimal for Snapdragon 865 / Android 10)
if(NOT DEFINED ANDROID_PLATFORM)
    set(ANDROID_PLATFORM android-29)
endif()

# Target ABI - Snapdragon 865 is arm64-v8a
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI arm64-v8a)
endif()

include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
