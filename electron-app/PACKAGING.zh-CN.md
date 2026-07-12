# Whispera Windows 打包说明

本文档对应当前推荐的便携包发布方式：

1. 主程序只生成 `win-unpacked/` 目录
2. 大模型和资源继续放在安装目录下的 `assets/`
3. 部分核心 Python 代码先编译成 `.pyd` 再随 Electron 一起分发，`voxcpm-tts-streaming-module` 仍以源码形式随包分发

## 当前推荐的交付结构

```text
Whispera/
  Whispera.exe
  assets/
    llama-bin/
    asr/
      SenseVoiceSmall/
    tts/
      openbmb__VoxCPM2/
    llm/
    lora/
    reference/
  resources/
    app-bundle/
      runtime/
        python/
      model/
        vad/
          silero_vad.onnx
      realtime/
        app.py
        *.pyd
      llm-module/
        scripts/
          start_llama_server.py
        src/
          llm_module/
            __init__.py
            *.pyd
      voxcpm-tts-streaming-module/
        src/
          voxcpm/
            *.py
            *.pyi
            *.json
```

说明：

- `realtime/app.py` 只保留一个很薄的启动器，用于 `python -m realtime.app`
- `realtime/` 和 `llm-module/src/llm_module/` 的业务逻辑放在编译后的 `.pyd` 里
- `voxcpm-tts-streaming-module/src/voxcpm/` 目前**仍然以 `.py` 源码形式打包**（PyTorch 模型代码 Cython 化目前不划算）
- `llm-module/scripts/start_llama_server.py` 保留 `.py`，用于 Electron 侧拉起 `llama-server`
- `model/vad/silero_vad.onnx` 跟随主包一起分发，不放到 `assets/`
- 前端仍然走 `asar`
- 这只做到把 `realtime` 和 `llm_module` 的源码从发布包里隐藏，不是完整的防逆向方案

## 为什么要先编译核心 Python

早期的打包配置会把 `realtime/`、`llm-module/`、`voxcpm-tts-streaming-module/` 三个目录整体原样拷进发布包，用户打开 `resources/app-bundle/` 就能直接看到全部 `.py` 源码。

现在的策略是：

- 开发态：继续使用仓库里的原始源码
- 发布态：先生成 `build/compiled-backend/`
- Electron 打包时只复制 `build/compiled-backend/`，不再复制原始源码目录
- 其中 `realtime` 和 `llm_module` 转成 `.pyd`；`voxcpm` 因为大量 PyTorch 动态特性，暂时保持 `.py` 拷贝

## 打包前提

需要先准备：

1. 可运行的 Python runtime
2. `Cython`
3. `electron-app/node_modules`

如果你使用当前仓库导出的 runtime，建议在那个环境里安装 `Cython`。

## 步骤 1：准备 runtime

```powershell
.\scripts\pack_runtime.ps1 -ReplaceExisting
```

如需再做一次瘦身：

```powershell
.\scripts\slim_runtime_for_distribution.ps1 -RuntimeRoot runtime\python
```

> 顺序提醒：瘦身会删掉 runtime 里的 `.lib`、`.pdb` 等文件。而步骤 2 编译 `.pyd` 时需要 `runtime\python\libs\python310.lib` 这个导入库。因此推荐 **先跑步骤 2 编译，再跑瘦身**；如果先瘦身，`slim_runtime_for_distribution.ps1` 已经把 `python310.lib`、`python3.lib`、`_tkinter.lib` 加入保护名单不会误删，但仍建议按“编译 → 瘦身”的顺序走，避免踩坑。

## 步骤 2：编译核心 Python 模块

```powershell
.\scripts\build_compiled_backend.ps1
```

如需显式指定带 `Cython` 的 Python：

```powershell
.\scripts\build_compiled_backend.ps1 -PythonExe C:\Users\you\anaconda3\envs\your_env\python.exe
```

默认输出目录：

```text
build/compiled-backend/
```

这个步骤会：

- 编译 `realtime` 里的核心模块为 `.pyd`
- 编译 `llm-module/src/llm_module` 为 `.pyd`
- 把 `voxcpm-tts-streaming-module/src/voxcpm` 里的 `.py`、`.pyi`、`.json` 原样拷贝过去（不做编译）
- 保留 `realtime/app.py`、`realtime/__init__.py`、`llm_module/__init__.py` 等最小必要的薄启动器和包结构文件
- 保留 `llm-module/scripts/*.py`（`start_llama_server.py` 需要 Electron 直接调用）

如果哪天你确认要让 `voxcpm` 也走 `.pyd`，把 [scripts/build_compiled_backend.py](../scripts/build_compiled_backend.py) 里 `SOURCE_ONLY_TREES` 那条挪进 `TARGETS`，并同步去掉 [electron-app/scripts/ensure_compiled_backend.js](./scripts/ensure_compiled_backend.js) 里 `forbiddenCompiledDirs` 对 voxcpm 的检查即可。目前不做，是因为 PyTorch 模型代码 Cython 化容易在导入或前向时炸。

编译完成后，脚本会额外验证一次：

```powershell
python -m realtime.app --help
```

这里使用的是 `build/compiled-backend` 作为 `PYTHONPATH` 的发布态结构验证。

## 步骤 3：构建便携目录产物

```powershell
cd electron-app
npm install
npm run dist
```

现在 `npm run dist` 会先检查：

- `build/compiled-backend/realtime/app.py` 是否存在
- `build/compiled-backend/llm-module/scripts/start_llama_server.py` 是否存在
- `realtime/` 和 `llm-module/src/llm_module/` 下是否至少存在一个 `.pyd`
- `voxcpm-tts-streaming-module/src/voxcpm/__init__.py` 是否存在
- `voxcpm-tts-streaming-module/src/voxcpm/` 下**不能**出现 `.pyd`（这个目录当前预期是纯源码分发）

如果没先跑编译脚本，会直接失败并提示，不会再悄悄把源码打进去。

当前 `dist` 只生成 `win-unpacked/` 目录，不再尝试生成 `NSIS Setup.exe`，也不再额外压缩成 zip。这是故意的，因为当前体积下压缩本身耗时明显，而实际交付时你完全可以按需手工处理。

## 当前会被打进主包的内容

- Electron 应用
- `resources/app-bundle/runtime/python/`
- `resources/app-bundle/model/vad/`
- `resources/app-bundle/realtime/`：`app.py` + `__init__.py` + 编译后的 `.pyd`
- `resources/app-bundle/llm-module/`：`scripts/*.py` + `src/llm_module/__init__.py` + 编译后的 `.pyd`
- `resources/app-bundle/voxcpm-tts-streaming-module/src/voxcpm/`：原始 `.py`、`.pyi`、`.json`（无 `.pyd`）
- `assets/` 目录骨架

## 当前不应打进主包的内容

- 原始 `realtime/*.py` 业务源码（`app.py`、`__init__.py` 除外）
- 原始 `llm-module/src/llm_module/*.py`（`__init__.py` 除外）
- `llm-module/llama/bin`
- `assets` 里的模型和外部资源

> voxcpm 目录当前是有意保留 `.py` 分发，见前文说明。

## 推荐交付物

当前推荐交付两部分：

1. `electron-app/dist/win-unpacked/`
2. 外部资源包，用户解压后放到同级 `assets/`

至少应包含：

- `assets/llama-bin/`
- `assets/asr/SenseVoiceSmall/`
- `assets/tts/openbmb__VoxCPM2/`
- `assets/lora/`

`assets/llm/` 依然可以选择：

- 半离线：不预放 `GGUF`，用户首次自己选本地模型
- 全离线：预放默认 `GGUF`

## 用户拿到后的使用方式

推荐给用户的步骤：

1. 把程序解压到长期保存的位置，例如 `D:\Apps\Whispera\`
2. 把资源包解压到同级 `assets/`
3. 启动应用
4. 如果 `assets/llm/` 里没有默认 `GGUF`，就在设置里手动选择
5. 在状态页做一次资源检测，确认 `Llama Runtime`、`ASR`、`TTS`、`GGUF` 状态正常

## 风险边界

这套方案的目标是：

- 不直接暴露 `realtime` 和 `llm_module` 的业务源码
- 维持现有便携包发布方式
- 尽量少改动开发态运行方式

它不代表：

- 代码绝对不可逆向（`.pyd` 只是 Cython 编译产物，静态分析仍然可以看）
- `voxcpm` 目录里的模型代码是隐藏的（当前是明文 `.py` 分发）
- 模型权重、LoRA、提示词策略天然安全

如果以后你希望进一步提升保护强度，可以继续往下面走：

1. 把 voxcpm 也纳入 Cython 编译，或改用 Nuitka / PyArmor 之类的方案
2. 把更多薄包装也改成原生可执行入口
3. 对前端 JS 继续压缩和混淆
4. 把最敏感的逻辑迁到远端服务
