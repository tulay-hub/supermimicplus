# 数据格式与打包内容

训练数据是 40 条基础动作与 113 条扩充动作，合计 153 条。独立测试参考 `cmu_91_04` 不在训练清单中；它来自 357 个符合初筛条件的去重候选随机抽样，随机种子 3623939731，选择后没有按策略结果重新挑选。

所有训练 CSV 为 50 Hz、28 列：`root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w, joint_0 ... joint_20`。关节顺序见 `src/unisupermimic/tasks/motion_tracking/lens110/supermimic/robot.py` 的 `JOINT_NAMES`。位置单位米、角度弧度。

发布时仅统一 CSV 注释头，数字文本逐字保留。文件哈希随注释变化，因此任务 YAML 和可迁移运行配置同步记录发布文件哈希；`numeric_text_sha256` 单独证明数字部分的一致性，模型权重保持原始字节。

`reports/dataset_manifest.json` 按以下组列出每个文件的帧数、频率、列数和 SHA-256：

- `base40`：第一阶段和第二阶段均使用的 40 条动作。
- `extension113`：第二阶段新增的 113 条动作。
- `heldout`：未参与训练的随机测试参考。
- `auxiliary`：早期单动作任务使用的辅助参考。
- `conversion_input40`：基础集转换前的 40 条输入，不作为 Lens110 训练参考。

数据包按原相对路径解压到 `src/unisupermimic/assets/motions/`，不需要旧工作区。完整资源清单和压缩包校验和见根目录 `release-manifest.json`。

逐帧数值、速度、限位、地面和自碰撞检查只能说明参考通过对应检查；动态可执行性仍需要策略回放。153 动作完整评测与单条未见动作结果保存在 `reports/`。
