# T3D-lyh 修改与验证手册

T3D（Trinity3D）根据平衡几何、粒子与能量源项及径向通量闭合，推进一维密度和温度剖面。此页列出维护版相对官方 v0.1.0 的累计修改，不把源码回归等同于实验验证。

- [源代码与版本](./SOURCE_VERSION.md)
- [在线手册](https://yihui-l.github.io/T3D-websites/Source-Code/)
- [官方 T3D 文档](https://t3d.readthedocs.io/en/latest/)

## 01 · 输运方程与数值求解

- **变步长 BDF2：** 时间系数只使用已接受的历史步长；拒步恢复剖面并重算 θ 法所需的旧时刻输运项，不把试探状态写入历史。 [源码](./T3D-lyh/t3d/Time.py) · [测试](./T3D-lyh/tests/unit/test_lyh_time_control.py)
- **时间步控制：** 校验时间参数与重试上限；首步和增步遵守 dtau_max，SI 输入只转换有量纲的时间量，不转换缩步比或误差阈值；到达终止时刻时消除仅由舍入产生的额外小步。 [源码](./T3D-lyh/t3d/Time.py)
- **Newton 求解：** 以差分形式构造时间离散残差并直接求解修正量，降低大项相减误差；Jacobian 用同一离散残差、受约束剖面和后端梯度响应，包含内部碰撞及热源导数。显式推进跳过 Jacobian。 [源码](./T3D-lyh/t3d/Solver.py) · [测试](./T3D-lyh/tests/unit/test_discrete_jacobian.py)
- **收敛诊断：** 分开记录 Newton 更新量、时间离散残差、半离散稳态速率；稳态停止看实际输运速率，不因某次 Newton 更新很小而停。时间离散残差诊断不改变默认 Newton 接受阈值。 [源码](./T3D-lyh/t3d/Solver.py) · [测试](./T3D-lyh/tests/unit/test_numerics_round3.py)
- **共享界面与磁轴：** 相邻控制体共用同一个物理粒子／热通量；磁轴通量严格为零，指定外侧剖面节点不参与推进或活动壳积分。 [源码](./T3D-lyh/t3d/Species.py) · [测试](./T3D-lyh/tests/unit/test_vmec_source_balance.py)
- **离散边界与终止：** 固定边界作为代数约束而非 BDF 外推；遇 stop 文件回滚未接受试探步；零时长或无演化变量时只写诊断，不强行推进。 [源码](./T3D-lyh/t3d/trinity_lib.py) · [测试](./T3D-lyh/tests/unit/test_driver_transactions.py)
- **径向网格与紧致梯度：** 修复三点网格的节点／界面插值和边界导数。紧致差分用等价带状求解；保留浮点精度并拒绝非法网格、剖面和零差分步。未测量加速比。 [源码](./T3D-lyh/t3d/Profiles.py) · [测试](./T3D-lyh/tests/unit/test_numerics_round3.py)
- **准中性与组分约束：** 一致处理从属物种、关联密度和温度剖面；支持显式选定准中性从属组分，拒绝重复组分、冲突约束和非正／非有限的密度或压强试探态。 [源码](./T3D-lyh/t3d/Species.py) · [测试](./T3D-lyh/tests/unit/test_species.py)

## 02 · 平衡与几何

- **VMEC 体积导数：** 从半网格 Nyquist (0,0) 雅可比系数取 dV/ds，再按 ρ=√s 计算 dV/dρ；径向节点和输运面共用一致的面积／体积度量。 [源码](./T3D-lyh/t3d/geometry_models/VMEC.py) · [测试](./T3D-lyh/tests/unit/test_vmec_source_balance.py)
- **磁面面积：** 周期角度采样不重复积分接缝；遇到不支持的非对称 wout 明确报错，不静默忽略非对称项。 [源码](./T3D-lyh/t3d/geometry_models/VMEC.py)
- **基本几何与重标度：** 圆形基本几何的 A/a² 和 a|∇ρ| 使用一致无量纲定义，修正磁场默认与 VMEC LCFS 回退；径向线平均不再带长度单位，非法度量明确报错。 [源码](./T3D-lyh/t3d/Geometry.py) · [测试](./T3D-lyh/tests/unit/test_driver_transactions.py)
- **GEQDSK／Miller：** 由实际能力选取唯一 GX 或 GS2-QL 几何提供者，不依赖自定义标签；GEQDSK 用 rminor 网格，逐面读入几何。Miller 必须启用兼容的导入，Boozer 预处理按后端能力执行。 [源码](./T3D-lyh/t3d/Geometry.py) · [测试](./T3D-lyh/tests/unit/test_gs2_ql_geometry_and_metric.py)
- **DESC 平衡反馈：** 在导入 JAX 后端前选定 CPU／GPU；压强增减均可触发，分开保存候选平衡，仅接受收敛结果；按需要输出 VMEC 格式。当前只接受不改变演化壳层体积／参考小半径的更新；没有移动磁面输运或压缩功。 [源码](./T3D-lyh/t3d/geometry_models/DESC.py) · [测试](./T3D-lyh/tests/unit/test_geometry_balance_interfaces.py)

## 03 · 物理源项与反馈

- **高斯源项归一化：** 对活动壳做体积积分；显式给定总源（包括零）按活动域归一化，不把固定外节点当作沉积单元；对域外中心的截断尾部给出警告，不把几乎无重叠误当作正常输入。 [源码](./T3D-lyh/t3d/Species.py) · [测试](./T3D-lyh/tests/unit/test_vmec_source_balance.py)
- **D–T 反应与 α 沉积：** 区分 α 出生功率、沉积热和聚变增益。关闭 α 热反馈或降低保留率不改变出生功率；独立 D/T 用两个 Maxwell 分布的相对温度。纯 H/e 不产生 D–T 功率。 [源码](./T3D-lyh/t3d/Physics.py) · [测试](./T3D-lyh/tests/unit/test_physical_recurrence.py)
- **碰撞／辐射：** 用碰撞成对能量交换校验输运方程与源项导数；辐射模型已含轫致辐射时避免重复计入；关闭源项时清除残留诊断。 [源码](./T3D-lyh/t3d/Physics.py) · [测试](./T3D-lyh/tests/unit/test_source_physics_io.py)
- **Zeff 与手动剖面：** 修复高斯／手动 Zeff 输入和混合电荷背景的杂质稀释，保持准中性；手工 n、T 表兼容 NumPy 2.x 并检查输入。 [源码](./T3D-lyh/t3d/Species.py) · [测试](./T3D-lyh/tests/unit/test_physical_recurrence.py)
- **辅助功率控制：** Q 控制使用出生聚变功率，把外部沉积计入增益分母，但仅调节可控的规定辅助功率；没有可调热源时不虚构反馈。 [源码](./T3D-lyh/t3d/Species.py) · [测试](./T3D-lyh/tests/unit/test_source_physics_io.py)
- **Sudo 与外侧边界：** 在初始化及已接受步之间应用密度反馈，单次通量、源项和残差计算使用同一个剖面。边界控制保持固定量、关联温度和准中性；发生跳变后下一步从一阶重启。Sudo 是经验反馈，不是额外演化方程。 [源码](./T3D-lyh/t3d/trinity_lib.py) · [测试](./T3D-lyh/tests/unit/test_driver_transactions.py)
- **PowerBalance：** 独立的源平衡通量闭合在所有源项收集后计算，不能作为额外通量叠加到其他闭合。 [源码](./T3D-lyh/t3d/flux_models/PowerBalance.py) · [测试](./T3D-lyh/tests/unit/test_geometry_balance_interfaces.py)
- **BEAMS3D：** 改用完整碰撞追踪；输入按 s、eV、m⁻³、kg 写入，输出用 √s 径向箱并汇总束线。修正 Fortran 起始下标、稀疏束线与同位素检查，分别控制粒子和能量源；可选独立 n,p 差分导数需要低标记噪声。目前离子沉积映射到主离子。 [源码](./T3D-lyh/t3d/source_models/BEAMS3D.py) · [测试](./T3D-lyh/tests/unit/test_beams3d_model.py)

## 04 · 通量模型与并行调度

- **ReLU：** 修复反向梯度的有符号通量；开启噪声时，同一步的基值与扰动共享抽样，不依赖全局随机种子，避免噪声直接污染有限差分。 [源码](./T3D-lyh/t3d/flux_models/ReLU.py) · [测试](./T3D-lyh/tests/unit/test_numerics_round3.py)
- **Rechester–Rosenbluth：** 在输运面使用同位置的参考归一化，补齐固定压强梯度下的密度扰动响应，不改原三次依赖。 [源码](./T3D-lyh/t3d/flux_models/RR.py) · [测试](./T3D-lyh/tests/unit/test_numerics_round3.py)
- **GX 闭合：** ReLU 代理保留有符号的湍流能量交换 H 及梯度；先用完整物种集构造受约束梯度扰动，再筛选动力学组分。修复逐半径／角度的 GEQDSK 读取，并检查单样本与非有限输出。 [源码](./T3D-lyh/t3d/flux_models/GX.py) · [测试](./T3D-lyh/tests/unit/test_backend_state_and_diagnostics.py)
- **GS2-QL：** 修复 GEQDSK 初始化和逐面读取；修正纯静电、零流剪切极限、单 k_y 取样以及有符号的组分权重。准线性模型见 Giacomin 等（2025）。 [源码](./T3D-lyh/t3d/flux_models/GS2_QL.py) · [测试](./T3D-lyh/tests/unit/test_gs2_ql_geometry_and_metric.py)
- **SFINCS：** 平衡文件拷贝完成后才启动求解；控制 MPI 秩与线程预算，检查 HDF5 完成标志、非线性收敛、形状及有限值，不把半写或失败输出当作有效通量。 [源码](./T3D-lyh/t3d/flux_models/SFINCS.py) · [测试](./T3D-lyh/tests/unit/test_backend_state_and_diagnostics.py)
- **GS2／NEO／KNOSOS 作业：** 用有界 CPU 调度和退出状态代替背景 shell 猜测；NEO 的输入解析器成功后才求解，KNOSOS 的径向输出必须齐全。 [源码](./T3D-lyh/t3d/flux_models/_external.py) · [测试](./T3D-lyh/tests/unit/test_external_transport_jobs.py)
- **GX GPU 与 MPI 调度：** 本机按设备分组、每个任务按 GPU 数申请 MPI 秩；修正 Polaris 三卡分配，支持 Slurm／自定义启动器限流和可选的无进展报错。调度测试未用真实 GPU 或 Slurm。 [源码](./T3D-lyh/t3d/flux_models/GX.py) · [测试](./T3D-lyh/tests/unit/test_external_transport_jobs.py)
- **拒步、缓存与失败清理：** 每次运行使用独立目录，步／迭代／拒步重试有独立标识；检查子进程退出和全部预期输出，终止仍在运行的兄弟作业。GX 仅把已成功并与同一几何匹配的检查点作热启动，不复用旧通量。 [源码](./T3D-lyh/t3d/trinity_lib.py) · [测试](./T3D-lyh/tests/unit/test_external_transport_jobs.py)

## 05 · 输入、重启与诊断输出

- **重启单位与网格：** 旧归一化量先还原到物理单位再用新参考量归一化，兼容旧 t_ref；拒绝坐标不匹配、缺测或掩蔽数据；变网格用壳层重叠映射，不声称跨几何体积守恒。 [源码](./T3D-lyh/t3d/Import.py) · [测试](./T3D-lyh/tests/unit/test_source_physics_io.py)
- **输入解析：** 保留实际 TOML／文件错误，不用错误的字符串回退覆盖原因；ASCII 多列源项求和，兼容 JETTO 时间键。 [源码](./T3D-lyh/t3d/Import.py) · [测试](./T3D-lyh/tests/unit/test_import.py)
- **检查点与输出路径：** 每 N 个接受步写检查点，NPY 和 NetCDF 写完才替换上一份，BP 单独路径；各格式不会共用同一个文件名。 [源码](./T3D-lyh/t3d/TrinityIO.py) · [测试](./T3D-lyh/tests/unit/test_driver_transactions.py)
- **诊断对齐与单位：** 修正逐物种和离子汇总、粒子量和积分单位，分开输出 α 出生与沉积功率；记录几何文件、时间推进阶数、残差与诊断有效性。 [源码](./T3D-lyh/t3d/TrinityIO.py) · [测试](./T3D-lyh/tests/unit/test_source_physics_io.py)
- **末帧与后处理：** 内部闭合在最终剖面重算；外部闭合如未同步则明确标 NaN，不为日志隐式启动求解。功率图按所选帧取通量，用已接受状态和已知阶数算时间导数，不猜旧文件的推进阶数。 [源码](./T3D-lyh/t3d/tools/source_sink_plot.py) · [测试](./T3D-lyh/tests/unit/test_driver_transactions.py)

## 06 · 官方案例：W7-X ReLU

固定密度、电子温度和外侧离子温度，只演化 (T_i)。两版各接受 8 步并到达 0.37405577 s；中间接受时刻不同。输入、VMEC 平衡与发行版参考逐字节核对。表内按各自终态剖面重算，未把官方输出中滞后的末帧通量当作终态。

| 终态指标 | 官方参考 | T3D-lyh |
| --- | ---: | ---: |
| 最内点离子温度（keV） | 1.42330319 | 1.40673474 |
| 演化域净离子热源（MW） | 1.403188 | 1.384315 |
| 最外输运面向外热流（MW） | 1.399212 | 1.382174 |
| BDF2 储能变化率（kW） | 3.98647 | 2.14071 |
| 功率平衡残差绝对值（W） | 10.85 | 0.117 |
| 内部界面未抵消量绝对值（W） | 74.53 | 0（舍入级） |

- 终态离子温度最大差 16.57 eV（最内点）。固定密度、电子温度和外侧离子温度一致。净离子热源来自规定电子剖面的碰撞能量交换。
- 残差为 `|dWi/dt − (Psource − Qout)|`。差异主要来自近轴体积权重和内部共享通量，其次有时间步轨迹和 Newton 收敛误差。结果不等价于实验预测精度提高。
- 整套远程测试 134 通过、4 跳过。真实 SFINCS 验证了 W7-X 单磁面、单 MPI 秩的启动、读回与归一化；之前也有四磁面和一组梯度扰动的粗分辨率测试。DESC 的已验证范围是 CPU 小平衡重求解，不是完整耦合输运。
- GX、GS2、BEAMS3D、KNOSOS、NEO、GPU/Slurm 和 ADIOS2 未完成真实运行验证；相应测试限于接口、合成数据或模拟调度。并行扩展性和移动磁面输运不在已验证范围内。

## 07 · 复现与参考

- 从 `Source-Code/T3D-lyh/` 启动，使用 `tests/regression/test-w7x-relu.in`。原始输入请求 NetCDF 和 BP；环境无 ADIOS2 时仅可在单独输入副本中关闭 `output_adios2`，不修改官方文件。
- 源码快照不含原始 Git 元数据。若安装器需显式版本，可在源码目录使用 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_T3D=0.1.0.post3 python -m pip install -e .`；该标签只用于打包，源码身份以 Git SHA 为准。
- [T3D 官方 W7-X ReLU 教程](https://t3d.readthedocs.io/en/latest/QuickReLU.html)
- [Barnes 等，TRINITY 隐式输运方法](https://doi.org/10.1063/1.3323082)
- [Giacomin 等，准线性闭合与 T3D](https://doi.org/10.1017/S0022377824001107)
- [Guttenfelder，T3D 集成模拟与功率诊断](https://doi.org/10.1017/S0022377825000339)
