#!/bin/bash
# Tutti 一键环境准备脚本
# 目标：在多种 Linux 发行版（Debian/Ubuntu/RHEL/CentOS/TencentOS/Fedora/openSUSE/Arch）
#       上自动安装 protobuf/gRPC/uuid 等依赖（gRPC 在 RHEL/TencentOS 等仓库缺失时回退 vcpkg）。

set -eu
# 注意：不开启 pipefail，因为 install_pkg 中允许部分非致命失败被吞掉。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- 运行环境探测 ----------------
ARCH="$(uname -m)"

# 计算 sudo 前缀：root 用户无需 sudo；非 root 但缺少 sudo 时给出明确错误。
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    echo "错误：当前为非 root 用户且未安装 sudo，无法继续安装系统依赖。" >&2
    echo "请使用 root 运行，或安装 sudo 后再试。" >&2
    exit 1
fi

# 解析命令行参数
# 默认使用全部 CPU 核数；nproc 不可用时回退到 4
if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS=4
fi

show_help() {
    cat <<EOF
用法: $0 [选项]
  -j N, --jobs N      并行编译线程数（用于 vcpkg 等后续编译，默认 ${JOBS} = nproc）
  -j=N, --jobs=N      同上
  -h, --help          显示帮助
环境变量:
  VCPKG_ROOT          指定 vcpkg 安装路径（默认 \${PROJECT_ROOT}/third_pkgs/vcpkg）
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -j|--jobs)
            if [ -z "${2:-}" ] || [[ "$2" == -* ]]; then
                echo "错误：$1 需要一个整数参数。" >&2
                exit 1
            fi
            JOBS="$2"
            shift 2
            ;;
        -j=*|--jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# 校验 JOBS 是正整数
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：--jobs 参数无效：$JOBS（必须为正整数）" >&2
    exit 1
fi

# 让后续所有 make / vcpkg 子流程默认并发
export MAKEFLAGS="-j${JOBS}"
export VCPKG_MAX_CONCURRENCY="${JOBS}"
echo "并行编译线程数 (JOBS): ${JOBS}"

# 检测包管理器并识别发行版家族
detect_pkg_manager() {
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_ID_LIKE="${ID_LIKE:-}"
    else
        DISTRO_ID="unknown"
        DISTRO_ID_LIKE=""
    fi

    if command -v apt-get >/dev/null 2>&1; then
        PKG_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
    elif command -v yum >/dev/null 2>&1; then
        PKG_MANAGER="yum"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MANAGER="zypper"
    elif command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
    else
        PKG_MANAGER="unknown"
    fi
    echo "检测到发行版: ${DISTRO_ID} (ID_LIKE=${DISTRO_ID_LIKE}), 包管理器: ${PKG_MANAGER}"
}

# 检查指定包是否已安装
pkg_is_installed() {
    local pkg="$1"
    case "$PKG_MANAGER" in
        apt)
            dpkg -s "$pkg" >/dev/null 2>&1
            ;;
        dnf|yum)
            rpm -q "$pkg" >/dev/null 2>&1
            ;;
        zypper)
            rpm -q "$pkg" >/dev/null 2>&1
            ;;
        pacman)
            pacman -Qi "$pkg" >/dev/null 2>&1
            ;;
        *)
            return 1
            ;;
    esac
}

# 安装包（自动按发行版选择名称）
# 第 4 个可选参数: "optional" 表示安装失败仅警告，不中断脚本
install_pkg() {
    local apt_name="$1"
    local rpm_name="$2"
    local pacman_name="$3"
    local optional="${4:-}"

    local target_name=""
    local install_cmd=""
    case "$PKG_MANAGER" in
        apt)
            target_name="$apt_name"
            install_cmd="$SUDO apt-get install -y $apt_name"
            ;;
        dnf)
            target_name="$rpm_name"
            install_cmd="$SUDO dnf install -y $rpm_name"
            ;;
        yum)
            target_name="$rpm_name"
            install_cmd="$SUDO yum install -y $rpm_name"
            ;;
        zypper)
            target_name="$rpm_name"
            install_cmd="$SUDO zypper install -y $rpm_name"
            ;;
        pacman)
            target_name="$pacman_name"
            install_cmd="$SUDO pacman -Sy --noconfirm $pacman_name"
            ;;
        *)
            echo "未识别的包管理器，请手动安装：$apt_name / $rpm_name"
            return 1
            ;;
    esac

    if pkg_is_installed "$target_name"; then
        echo "$target_name 已安装。"
        return 0
    fi

    echo "正在安装 $target_name ..."
    if [ "$PKG_MANAGER" = "apt" ]; then
        $SUDO apt-get update >/dev/null
    fi
    if eval "$install_cmd"; then
        return 0
    fi

    if [ "$optional" = "optional" ]; then
        echo "警告：$target_name 安装失败（该包在当前发行版可能不可用），将跳过；请关注后续依赖检查。"
        return 0
    fi
    return 1
}

detect_pkg_manager

echo "------------------------------------------------------------"
echo "  Tutti 环境准备脚本"
echo "  项目根目录 : ${PROJECT_ROOT}"
echo "  架构       : ${ARCH}"
echo "  特权方式   : ${SUDO:-(root 直接执行)}"
echo "  并行数     : ${JOBS}"
echo "  发行版     : ${DISTRO_ID} (ID_LIKE=${DISTRO_ID_LIKE})"
echo "  包管理器   : ${PKG_MANAGER}"
echo "------------------------------------------------------------"

# 安装 uuid 开发库
#   Debian/Ubuntu : uuid-dev
#   RHEL/CentOS/TencentOS/Fedora/SUSE : libuuid-devel
#   Arch          : util-linux (自带 libuuid)
install_pkg "uuid-dev" "libuuid-devel" "util-linux"

# 安装 wget / unzip（后续步骤需要）
install_pkg "wget"  "wget"  "wget"
install_pkg "unzip" "unzip" "unzip"

# 安装 Protobuf 开发库 + protoc 编译器（CMakeLists.txt: find_package(Protobuf REQUIRED)）
#   Debian/Ubuntu : libprotobuf-dev + protobuf-compiler
#   RHEL/CentOS/TencentOS/Fedora/SUSE : protobuf-devel + protobuf-compiler
#   Arch          : protobuf
install_pkg "libprotobuf-dev"   "protobuf-devel"    "protobuf"
install_pkg "protobuf-compiler" "protobuf-compiler" "protobuf"

# 安装 gRPC 开发库 + grpc_cpp_plugin（CMakeLists.txt: find_package(gRPC REQUIRED)）
#   Debian/Ubuntu : libgrpc++-dev + protobuf-compiler-grpc
#   RHEL/CentOS/TencentOS/Fedora     : grpc-devel + grpc-plugins (EL8 系仓库均无)
#   SUSE          : grpc-devel
#   Arch          : grpc
# 系统包不可用时（典型场景：TencentOS 3 / EL8），fallback 到 vcpkg 编译安装。
install_pkg "libgrpc++-dev"          "grpc-devel"   "grpc" optional
install_pkg "protobuf-compiler-grpc" "grpc-plugins" "grpc" optional

# ----- 通过 vcpkg 提供 C++ 依赖（grpc / yaml-cpp 等） -----
# 触发逻辑：若系统未提供 gRPC（通过 pkg-config 与 grpc_cpp_plugin 检测），
# 则启动 vcpkg 来安装 grpc + yaml-cpp 等需要的 C++ 库。
# 之所以把 yaml-cpp 一并放进 vcpkg：
#   - RHEL/TencentOS 系统包仅 yaml-cpp 0.5.x，其 CMake config 缺少
#     RelWithDebInfo 配置的 IMPORTED_LOCATION，触发 CMP0111 警告刷屏。
#   - 由 vcpkg 统一安装可保证版本一致、CMake 集成干净。
ensure_cpp_deps_via_vcpkg() {
    if command -v pkg-config >/dev/null 2>&1 \
        && pkg-config --exists grpc++ 2>/dev/null \
        && command -v grpc_cpp_plugin >/dev/null 2>&1; then
        echo "系统已提供 gRPC（grpc++ + grpc_cpp_plugin），跳过 vcpkg 安装。"
        return 0
    fi

    echo "系统未提供 gRPC，回退到 vcpkg 安装方案。"

    # 代理环境健康提示（vcpkg 经常因公司代理把 GitHub release 拦掉）
    if [ -n "${HTTPS_PROXY:-}" ] || [ -n "${https_proxy:-}" ]; then
        echo "提示：检测到 HTTPS 代理：${HTTPS_PROXY:-${https_proxy:-}}"
        echo "      如果 vcpkg 编译期出现 squid/HTML 错误页，多半是代理拦截了某个下载源；"
        echo "      可临时取消代理重试: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy"
    fi

    # vcpkg 编译 grpc 需要的构建依赖
    # 说明：build-essential / gcc-c++ / base-devel 元包通常已包含 make 与 tar，
    #       因此不再单独安装 make / tar 以避免冗余。
    install_pkg "build-essential" "gcc-c++"     "base-devel"
    install_pkg "cmake"           "cmake"       "cmake"
    install_pkg "git"             "git"         "git"
    install_pkg "pkg-config"      "pkgconf-pkg-config" "pkgconf" optional
    install_pkg "curl"            "curl"        "curl"
    install_pkg "zip"             "zip"         "zip"
    # autotools / perl 是 grpc 的若干传递依赖（abseil/c-ares/openssl 等）需要
    install_pkg "autoconf"        "autoconf"    "autoconf" optional
    install_pkg "automake"        "automake"    "automake" optional
    install_pkg "libtool"         "libtool"     "libtool"  optional
    install_pkg "perl"            "perl"        "perl"     optional

    local vcpkg_root="${VCPKG_ROOT:-${PROJECT_ROOT}/third_pkgs/vcpkg}"
    if [ ! -d "$vcpkg_root/.git" ]; then
        echo "克隆 vcpkg 到: $vcpkg_root"
        git clone --depth 1 https://github.com/microsoft/vcpkg.git "$vcpkg_root" || {
            echo "vcpkg 克隆失败，请检查网络。"
            return 1
        }
    else
        echo "vcpkg 已存在: $vcpkg_root"
    fi

    if [ ! -x "$vcpkg_root/vcpkg" ]; then
        echo "bootstrap vcpkg ..."
        bash "$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics || {
            echo "vcpkg bootstrap 失败。"
            return 1
        }
    fi

    # 通过 vcpkg 安装一个包；已安装则跳过
    # 用法：vcpkg_install_pkg <port-name>
    vcpkg_install_pkg() {
        local port="$1"
        if "$vcpkg_root/vcpkg" list 2>/dev/null | grep -q "^${port}:"; then
            echo "vcpkg 已安装 ${port}。"
            return 0
        fi
        echo "通过 vcpkg 安装 ${port}（使用 ${JOBS} 个并发任务）..."
        "$vcpkg_root/vcpkg" install "${port}" || {
            echo "vcpkg 安装 ${port} 失败。"
            return 1
        }
    }

    # 需要通过 vcpkg 提供的 C++ 库
    #   grpc     : NVMeService（系统包通常不可用）
    #   yaml-cpp : 配置文件解析（系统包在 RHEL/TencentOS 上仅 0.5.x，
    #              缺少 RelWithDebInfo 的 IMPORTED_LOCATION，会产生大量 CMP0111 警告）
    local vcpkg_ports=("grpc" "yaml-cpp")
    for port in "${vcpkg_ports[@]}"; do
        vcpkg_install_pkg "$port" || return 1
    done

    # 二次校验：关键产物存在，避免代理/网络异常导致的"伪成功"
    local grpc_cmake_dir="${vcpkg_root}/installed/x64-linux/share/grpc"
    local grpc_plugin_bin="${vcpkg_root}/installed/x64-linux/tools/grpc/grpc_cpp_plugin"
    if [ ! -f "${grpc_cmake_dir}/gRPCConfig.cmake" ]; then
        # 兼容其它 triplet（如 arm64-linux）
        if ! find "${vcpkg_root}/installed" -maxdepth 4 -name 'gRPCConfig.cmake' 2>/dev/null | grep -q .; then
            echo "错误：vcpkg 报告安装成功，但找不到 gRPCConfig.cmake，疑似下载/编译异常。"
            echo "可执行 '${vcpkg_root}/vcpkg remove grpc --recurse' 后重试。"
            return 1
        fi
    fi
    if [ ! -x "$grpc_plugin_bin" ]; then
        if ! find "${vcpkg_root}/installed" -maxdepth 5 -name 'grpc_cpp_plugin' -type f -executable 2>/dev/null | grep -q .; then
            echo "错误：找不到 grpc_cpp_plugin 可执行文件。"
            return 1
        fi
    fi
    if ! find "${vcpkg_root}/installed" -maxdepth 4 -name 'yaml-cpp-config.cmake' 2>/dev/null | grep -q .; then
        echo "警告：未找到 yaml-cpp-config.cmake，可能 vcpkg 中的 yaml-cpp 安装异常。"
    fi
    echo "vcpkg 中 gRPC / yaml-cpp 安装校验通过。"

    VCPKG_TOOLCHAIN_FILE="${vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
    cat <<EOF

==============================================================
gRPC 已通过 vcpkg 安装到: ${vcpkg_root}

推荐使用 CMake Presets（一行搞定）：

    cmake --preset default
    cmake --build --preset default

可用 preset：default / debug / release / system
查看完整列表：cmake --list-presets

如果你的 CMake < 3.21 不支持 presets，可手动指定 toolchain：

    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=${VCPKG_TOOLCHAIN_FILE}
==============================================================
EOF
    return 0
}

ensure_cpp_deps_via_vcpkg || {
    echo "警告：通过 vcpkg 安装 C++ 依赖（grpc / yaml-cpp）失败，请参考 README 手工处理。"
}

# 校验 protoc / grpc_cpp_plugin 是否真的可用
if ! command -v protoc >/dev/null 2>&1; then
    echo "警告：protoc 未在 PATH 中，CMake 阶段会失败。请确认 protobuf 编译器已正确安装。"
fi
if ! command -v grpc_cpp_plugin >/dev/null 2>&1; then
    if [ -n "${VCPKG_TOOLCHAIN_FILE:-}" ]; then
        echo "提示：grpc_cpp_plugin 通过 vcpkg 安装，CMake 会自动从 toolchain 找到，无需在 PATH 中。"
    else
        echo "警告：grpc_cpp_plugin 未在 PATH 中。若使用系统 grpc，请确认 grpc-plugins / protobuf-compiler-grpc 已安装。"
    fi
fi

echo "操作完成！"