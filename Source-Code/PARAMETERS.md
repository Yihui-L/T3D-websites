# T3D-lyh 完整输入参数

适用于维护版 d5ac1fe。表中键名区分大小写；省略表示不写该键，而不是写 TOML null。默认值和约束以维护分支实际代码为准；参数可读不等于相关外部环境均已实跑。物理公式及 I/O 见[源码手册](https://yihui-l.github.io/T3D-websites/Source-Code/)；计算示例见 [W7-X ReLU](https://yihui-l.github.io/T3D-websites/W7X-ReLU-lyh/)。

本表逐项对应 `T3D-lyh d5ac1fe` 主驱动所读取的 TOML 配置。默认等于省略该项的结果；**未设**表示应省略，不应在 TOML 中写 `null`。布尔下文均取 `true` / `false`；除另注外，文件路径按运行时工作目录解析。`[[species]]`、`[[model]]`、`[[source]]` 是数组表，`[import]` 等是单表。后端模板自己的 TOML 或 Fortran namelist 不属于 T3D 顶层参数。

## 径向网格 `[grid]`

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `N_radial` | 10 | 整数 ≥ 3 | 剖面节点数；含最外 Dirichlet 节点，活动壳层和中点通量面各有 N−1 个。 |
| `rho_edge` | 0.8 | 0 < ρ ≤ 1，无量纲 | 外侧剖面节点坐标，不一定是 LCFS 或最后一个活动界面。 |
| `flux_label` | `"torflux"` | `"torflux"`、`"rminor"` | 分别表示归一化环向磁通平方根和几何小半径比 r/a；影响几何、文件、模型之间的径向映射。 |
| `rho_inner` | ρ_edge/(2N−1) | 历史键，给值会警告 | 当前不能手动配置；内侧剖面节点由均匀网格决定，磁轴零通量独立施加。 |

## 时间推进 `[time]`

默认时间单位是参考时间 t_ref。`[time].alpha` 是时间离散权重，与剖面参数 `alpha` 无关。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `dtau` | 0.5 | 有限正数，s 或归一化时间 | 初始步长；先限制到 dtau_max，最后一步可裁剪到 t_max。 |
| `dtau_max` | 10.0 | 有限正数，s 或归一化时间 | 步长上限。 |
| `dtau_adjust` | 2.0 | 有限实数 > 1，无量纲 | 拒步后的除数；合格接受步的增长倍率另固定为 2。 |
| `dtau_increase_threshold` | newton_threshold/4 | 非负有限实数，无量纲 | 接受步的 RMS 足够小才允许增长步长。 |
| `N_steps` | 1000 | 整数 ≥ 0 | 最大接受步数；0 时不推进。 |
| `t_max` | 1000 | 非负有限实数，s 或归一化时间 | 终止时间；0 时不推进，仍可写诊断。 |
| `alpha` | 1 | 有限实数 [0,1] | θ 隐式权重。0 显式，0 < α < 1 为 θ 方法；1 首步后向 Euler，之后用接受历史做变步长 BDF2。 |
| `max_newton_iter` | 4 | 整数 ≥ 1 | 每个尝试步允许的 Newton 迭代次数。 |
| `newton_threshold` | 0.02 | 非负有限实数，无量纲 | 常规 Newton 接受阈值（活动节点的相对剖面更新量 RMS；隐式残差另记诊断，不参与此阈值）。 |
| `newton_tolerance` | 0.1 | 非负有限实数，无量纲 | 达到迭代上限时允许的较宽接受阈值。 |
| `rms_converged` | 1e−16 | 非负有限实数，归一化空间变化率阈值 | 按各活动节点的相对空间平衡残差判断是否提前稳态终止；0 禁用，不等于 Newton 容差。 |
| `max_timestep_retries` | 20 | 整数 ≥ 0 | 同一已接受时刻还能重试的次数上限。 |
| `use_SI` | `false` | 布尔 | 为 true 时 dtau、dtau_max、t_max 为秒；只换算这三个量，倍率、RMS 阈值不换算。 |

## 平衡几何 `[geometry]`

VMEC、DESC 需 `torflux`；Miller 和 GEQDSK 需 `rminor`。GEQDSK 还需恰好一个 GX 或 GS2‑QL 几何提供者。输入的几何数据必须与其他径向数据的磁通标签一致。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `geo_option` | `"basic"` | `"basic"`、`"vmec"`、`"desc"`、`"geqdsk"`、`"miller"` | 圆形近似、VMEC wout、DESC 平衡、GEQDSK 或由读入文件提供的 Miller 几何。 |
| `geo_file` | 空字符串 | 文件路径 | VMEC 用 wout NetCDF；DESC 用原生平衡文件（通常 .h5）或 VMEC .nc；GEQDSK 用平衡文件，由后端读。 |
| `AspectRatio` | 0（basic 回退 4） | 正实数 R/a，无量纲 | basic 可设；VMEC、DESC、GEQDSK 由平衡确定；Miller 从输入数据确定。 |
| `iota23` | 0.75 | 正实数，无量纲 | 用于 ISS04 等的 ρ≈2/3 旋转变换；相关读入平衡可覆盖此值。 |
| `import` | `false` | 布尔 | Miller 必须启用，且共享 [import] 必须提供兼容几何；不用于替代 VMEC/ DESC 的 geo_file。 |
| `evolve_equilibrium` | `false` | 布尔；启用只支持 `desc` | 允许按输运压强触发重算。不能把固定几何的回归理解成已验证移动磁面守恒。 |
| `update_equilibrium` | 未设 | 布尔，历史别名 | 仅在 [geometry] 下等价于上项。两者同时出现且冲突会报错；[physics].update_equilibrium 不被读，官方旧算例保留该无效键。 |
| `desc_pressure_delta_threshold` | 1.25 | 启用时有限实数 ≥ 1 | 最大双向压强比超过该阈值才更新平衡。 |
| `desc_pressure_delta` | 2 | 启用时有限实数 ≥ 1 | 更新触发后，小于此双向压强比走扰动，大变化走完整求解。 |
| `desc_use_vmec_format` | 按文件后缀推断 | 布尔；保留未设可自动推断 | 控制 DESC 重算的输出格式及依赖 VMEC 格式的耦合路径，不改变输入读取器；.nc 时默认 true，原生格式默认 false。 |
| `desc_device` | `"auto"` | `"auto"`、`"cpu"`、`"gpu"` | 在导入计算后端前选择 JAX 设备；auto 优先可用 GPU。 |
| `geo_outputs` | `"geo/"` | 目录路径 | DESC 更新的平衡和日志目录，各次求解在隔离子目录。 |
| `use_beta_prime_alpha` | `false` | 布尔 | GS2-QL 用读入的 Miller β′_α 作为几何输入。 |
| `beta_prime_const` | `false` | 布尔 | 用初始或导入的 β′，而不是每次随剖面更新（现行 GS2-QL 分支）。 |
| `force_beta_prime_species` | `false` | 布尔 | 与 beta_prime_const 配合，沿用支持的 Miller 重启/几何所提供的 β′；不是所有几何均有该数据。 |
| `compute_surface_areas` | `true` | 布尔，兼容键 | 当前版本读取但无后续控制分支，不会关闭通量面面积计算。 |
| `overwrite` | `false` | 布尔，兼容键 | 当前 Geometry 只保存该键；不控制求解器后端各自的 overwrite。 |

## 尺寸与磁场缩放 `[rescale]`

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `L_mult` | 1 | 推荐有限正实数，无量纲 | 线性尺度倍率；0 为历史输入，回退 1。体积随尺度的立方变化（适用的几何分支）。 |
| `L_def` | 空格（自动） | `"minor"`、`"major"` 或省略 | basic/VMEC 选择小/大半径为标度。VMEC 指定模式时须给对应的 a_minor 或 R_major；DESC、GEQDSK 走 a_minor 优先逻辑。 |
| `a_minor` | 0（未覆盖） | 正实数，m | 小半径标度，最终乘 L_mult；basic 若强制 minor 却为 0 回退 1 m。 |
| `R_major` | 0（未覆盖） | 正实数，m | 大半径标度，乘 L_mult；basic 走大半径而未给值时回退 4 m；VMEC 的 major 模式须给正值。 |
| `B_mult` | 1 | 推荐有限正实数，无量纲 | 磁场倍率；0 在常用几何分支回退 1。 |
| `B_def` | 空格（自动） | `"LCFS"`、`"volavgB"`、`"vmecB0"` 或省略 | basic/VMEC 的磁场规格。DESC、GEQDSK 主要使用 Ba 覆盖加倍率。 |
| `Ba` | 0（未覆盖） | 正实数，T | 边缘/参考磁场输入；VMEC 的 LCFS 模式可覆盖读入值，basic 默认 1 T。 |
| `volavgB` | 0（未覆盖） | 正实数，T | B_def=volavgB 时令体平均磁场达到给定标度；basic 将其作为 Ba。 |
| `vmecB0` | 0（未覆盖） | 正实数，T | B_def=vmecB0 时令轴上磁场达到给定标度；不是把 Btor 直接设为 B0。 |

## 组分 `[[species]]`

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `type` | `"deuterium"` | 已知：`electron`、`hydrogen`、`deuterium`、`tritium`、`helium`、`boron`、`carbon`、`oxygen`、`neon`、`krypton`、`tungsten`，或带 mass 和 Z 的自定义离子名 | 不同组分名称必须唯一，须含电子和至少一个离子。 |
| `tag` | 内置 e、H、D、T、He、B、C、O、Ne、Kr、W 等，否则用 type | 字符串 | 日志、输出变量和文件导入缺省字段后缀。 |
| `mass` | 内置组分表 | 正有限实数，m_p | 质量乘质子质量；自定义离子显式提供。 |
| `Z` | 内置组分表 | 电子 −1、离子有限正数，e | 电荷数，参与准中性、碰撞与有效电荷。 |
| `adiabatic` | `false` | 布尔，至多一个物种 | 标记绝热响应并关闭该组分独立的密度和温度推进；仅固定状态不等于开启它。 |
| `bulk` | `false`（无标记则首个离子） | 布尔，至多一个且不能为电子 | 主离子选择。 |
| `use_as_reference` | 随 bulk（无显式选择则首个离子） | 布尔，至多一个 | 决定湍流/gyro-Bohm 归一化参考组分。 |
| `impurity` | 是否给定该组分的 Zeff 目标 | 布尔，至多一个离子 | 将该物种标为初始化杂质，参与准中性从属组分选择。 |
| `Zeff` | 未设 | 实数，满足 Z_eff,background < 目标 < Z_impurity | 初始化指定有效电荷并按电荷比例稀释其余离子；不同于 [physics].Zeff 的制动辐射系数设置。 |
| `qneut_species` | `false`（自动挑选从属根） | 布尔；至多一个，不可再给 density.equal_to | 此组分的密度由逐面准中性计算，而不是独立推进；当现有链接已恒等满足准中性时不必消去自由度。 |
| `rel_step` | 0.5 | 非负有限实数 | 通量闭合的相对梯度探测步幅。 |
| `abs_step` | 0.5 | 非负有限实数 | 通量闭合的绝对梯度探测下限；有效探测步 max(rel_step×基准梯度绝对值, abs_step) 必须非零。 |
| `density`、`temperature` | 空表 | 内联表，见下 | 分别设置粒子密度和温度的初值/边界/演化自由度。 |
| `aux_particle_source`、`aux_power_source` | 空表 | 内联表，见下 | 分别设置辅助加料和加热的体源。 |
| `density_source`、`pressure_source` | 未设 | 历史别名 | 与对应新键同现时旧表覆盖新表；建议只使用新键。 |
| `density_relu_flux`、`pressure_relu_flux` | 空表 | 内联表，见下 | ReLU 的组分级粒子/热通量参数；后者虽名 pressure，阈值作用在温度梯度。 |
| `particle_flux`、`heat_flux` | 空表 | 内联表，见下 | 为 Import 模型绑定指定的穿面粒子率和热功率。 |

### 状态剖面 `density`、`temperature`

状态单位：n 用 10²⁰ m⁻³，T 用 keV。表内键在 `density` 和 `temperature` 中各自使用；组分间链接在初始化与推进中分开处理。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `evolve` | `true` | 布尔 | 是否作为独立未知量。准中性、绝热或 equal_to 从属物种可移除此自由度。 |
| `equal_to` | 未设 | `"electron"` 或 `"0.5 * electron"`；系数有限且 > 0 | 全程保持与目标组分同类剖面的比例，不独立推进；拒绝循环。 |
| `init_to` | 未设 | 同 equal_to | 只在初始化按比例设置，之后按本组分 evolve 状态处理；同一剖面不可同时设 equal_to。 |
| `shape` | `"parabolic"` | `"parabolic"`、`"gaussian"`、`"exponential"`、`"manual"`、`"custom"` | 内置初始剖面表达；启用 import 时文件读入优先。 |
| `core` | 密度 4；温度 8 | 对应状态单位 | parabolic 默认核心幅值。 |
| `edge` | 未设（parabolic 回退：密度 0.5、温度 2） | 对应状态单位 | 外侧约束；读文件时若显式指定会覆盖最外径向点。指数及自定义分支具有各自的重标定逻辑。 |
| `sep` | 未设 | 对应状态单位 | parabolic 给非零 core 和 sep 时以 ρ=1 为参考边界；低于 lineavg/volavg 优先级。 |
| `lineavg` | 未设 | 对应状态单位 | parabolic/custom 目标算术节点平均，实现中为离散节点平均，不是合成诊断弦积分。 |
| `volavg` | 未设 | 对应状态单位 | parabolic/custom 目标体平均；parabolic 在没有有效 lineavg 时使用。 |
| `alpha` | 1 | 有限实数；平滑正剖面通常取正 | parabolic 的外幂 α。 |
| `alpha1` | 2 | 有限实数；平滑正剖面通常取正 | parabolic 的径向内幂；形状为 (1−x^alpha1)^alpha。 |
| `height` | 0 | 对应状态单位 | gaussian 峰值；须配合有效 integrated 或非零正幅值，保证初值正。 |
| `integrated` | 未设 | 状态单位 × m³ | gaussian 体积分归一化目标；非零时优先于 height。 |
| `width` | 0.1 | 正实数，ρ 的单位 | gaussian 标准差。 |
| `center` | 0 | 有限实数，ρ 的单位 | gaussian 中心。 |
| `coefs` | [0,0,0,0] | 数字数组，使用前四项；不足补 0 | exponential 的 c₀…c₃：c₀ exp(−c₁ρ²−c₂ρ⁴−c₃ρ⁶)；c₀ 用状态单位。 |
| `import` | `false` | 布尔 | 从共享 [import] 读初始状态；是否演化另由 evolve 决定。 |
| `key` | 未设 | 变量/列名字符串（单数） | 文件字段；ASCII/T3D 默认 n_<tag> 或 T_<tag>；JETTO 必须显式指定。 |
| `profstr` | 未设 | 两行数字字符串，以分号隔开半径和值；数值可逗号或空白分隔 | manual 用 PCHIP；横坐标是 ρ/ρ_edge，严格递增并覆盖节点，剖面值有限且正。 |
| `func` | 未设 | 含 `def init(rho): ...` 的多行 Python 字符串 | custom 参数；输入代码由本进程执行，仅使用可信 TOML；函数返回与网格长度匹配的值或标量。 |

### 辅助源 `aux_particle_source`、`aux_power_source`

静态导入是一次性快照；没有文件时序强迫。高斯归一化使用演化域，不把固定边界节点当活动壳层。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `shape` | `"gaussian"` | 目前仅 `"gaussian"` | 不导入文件时的径向形状；代码求活动壳层平均。 |
| `height` | 0 | 有限实数；粒子 10²⁰ m⁻³ s⁻¹、热 MW m⁻³ | 规定体源峰值；integrated 未设才使用，允许有符号源或汇。 |
| `integrated` | 未设 | 有限实数；粒子 10²⁰ s⁻¹、热 MW | 以活动体积分为目标；显式 0 会覆盖 height 并给零源。分布太窄或大部分在域外会警告或拒绝。 |
| `width` | 0.1 | 正有限实数，ρ 的单位 | 非零高斯的标准差。 |
| `center` | 0 | 有限实数，ρ 的单位 | 非零高斯的中心。 |
| `import` | `false` | 布尔 | 使用共享 [import] 的体源列/场，替代高斯，不会关闭 collisions 等独立物理源；只进对应演化方程。 |
| `keys` | 未设 | 字符串或字符串数组，取决于读者 | ASCII、TRANSP、JETTO 可对多项求和；T3D 重启指定一个源诊断字段，读取后转换归一化，默认 Sn_aux_<tag> 或 Sp_aux_<tag>。 |

### ReLU 闭合与规定通量绑定

ReLU 的模型级参数与外部求解器参数列于下一组模型表。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `density_relu_flux.critical_gradient` | 1.0 | 实数，无量纲 | 粒子通量仿射零点 κc,0 的旧式名字；不是粒子 hard-ReLU 截止。 |
| `density_relu_flux.critical_gradient0` | 采用 critical_gradient | 实数，无量纲 | 覆盖粒子分支零点截距。 |
| `density_relu_flux.critical_gradient1` | 0.0 | 实数，无量纲 | 粒子零点沿 ρ 的线性变化系数。 |
| `density_relu_flux.slope` | 0.5 | 实数；非负为常见物理选择 | 粒子分支幅值；0 关闭该分支。 |
| `density_relu_flux.noise_frac` | 0.0 | 兼容键 | 当前读取但不用于 ReLU；使用模型级 noise。 |
| `pressure_relu_flux.critical_gradient` | 2.0 | 非负阈值，无量纲 | 热分支温度对数梯度阈值截距的旧式名字。 |
| `pressure_relu_flux.critical_gradient0` | 采用 critical_gradient | 实数，无量纲 | 显式截距覆盖旧键；应与梯度系数一起保证所有活动面上的阈值非负。 |
| `pressure_relu_flux.critical_gradient1` | 0.0 | 实数，无量纲 | 热分支阈值沿 ρ 的线性变化系数；κc=κc,0+ρ κc,1。 |
| `pressure_relu_flux.slope` | 0.5 | 实数；非负为常见物理选择 | 热分支幅值；0 关闭该分支。 |
| `pressure_relu_flux.slope0` | 采用 slope | 兼容键 | 当前读取但未参与 ReLU 求值。 |
| `pressure_relu_flux.slope1` | 0.0 | 兼容键 | 当前读取但未实现组分斜率随半径变化。 |
| `pressure_relu_flux.noise_frac` | 0.0 | 兼容键 | 当前读取但不用于 ReLU；使用模型级 noise。 |
| `particle_flux.import`、`heat_flux.import` | `false` | 布尔；需启用 `[[model]] model="Import"` | 选择固定物理穿面率；粒子限独立 n 方程，热限独立 p 方程。 |
| `particle_flux.keys`、`heat_flux.keys` | `Gam_SI20_<tag>`、`Q_MW_<tag>` | ASCII 单个列名或不同列名数组；T3D 单个有量纲诊断名字 | ASCII 列相加；分别为 10²⁰ s⁻¹ 与 MW，正向外。不要把通量密度或 GB 通量当成穿面率，在所有活动面覆盖范围内按面插值，不外推。 |

## 内置物理 `[physics]`

历史 `[debug]` 仅在未给 `[physics]` 时作为它的整表别名，而不与 `[physics]` 合并。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `collisions` | `true` | 布尔 | 启用物种间库仑能量交换；作用于 p 方程，不是径向新经典输运。 |
| `alpha_heating` | `true` | 布尔 | 启用 D–T α 沉积热源；出生与沉积分开诊断。 |
| `alpha_efficiency` | 1.0 | 实数 [0,1] | 出生 α 功率中进入模型局域能量源的份额。 |
| `radiation` | `false` | 布尔 | 启用杂质辐射（线、复合、轫致）；与下项同真时优先，不重复加上分离的轫致辐射项。 |
| `bremsstrahlung` | `false` | 布尔 | 单独启用基于 Z_eff 的轫致辐射。 |
| `bremstrahlung` | 继承 bremsstrahlung | 布尔，历史拼写 | 若存在会覆盖规范拼写；新文件不要使用。 |
| `turbulent_exchange` | `false` | 布尔 | 将支持闭合提供的湍流物种能量交换加入压力源；不把径向热流本身变成交换源。 |
| `f_tritium` | 0.5 | 实数 [0,1] | 没有单独氚剖面时，用现有氘密度按隐含 D/T 混合比例估算聚变。 |
| `f_helium` | 0 | 实数 [0,1] | 燃料稀释的现象学系数，使反应计算的 n_D 和 n_T 分别乘 (1−f_helium)；不是独立的氦粒子输运模型。 |
| `scale_density_to` | 未设 | `"sudo_line"`、`"sudo_vol"` 或 `"正系数 * sudo_line"` 等 | 按 Sudo 密度极限重标定目标线平均/体平均；启用时不得与外部加热后端合用。 |
| `Zeff` | 未设 → 由组分求 | 数值，或下列内联表 | 只规定该分支所用的有效电荷剖面；并非初始化杂质密度。仅在相关物理模型中使用。 |

### 轫致辐射使用的 `[physics].Zeff`

数值 `Zeff = 1.5` 直接设常量。若用表选择解析形式，不应把默认的 0 理解为物理推荐值。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `from_species` | 非空表为 false；空表/缺省为 true | 布尔 | 以 Σ Z_s² n_s / n_e 即时计算，优先于导入和预置形状。 |
| `import` | `false` | 布尔 | 从当前实现支持的 TRANSP U-file 读取 Z_eff；ASCII、T3D 或 JETTO 读者不提供该专用入口。 |
| `key` | 未设 | 变量名字符串 | TRANSP 默认依次用 ZEFFR / ZEFF。 |
| `shape` | `"parabolic"` | `"parabolic"`、`"gaussian"`、`"exponential"`、`"manual"` | 选择静态剖面，无 custom 模式。 |
| `core` | 0 | 实数；Z_eff 无量纲 | parabolic 核心幅值，应给物理合理的正值。 |
| `edge` | 0 | 实数；Z_eff 无量纲 | parabolic 外侧参数。 |
| `sep` | 0 | 实数；Z_eff 无量纲 | 非零时给 ρ=1 参考值，见 parabolic 分支。 |
| `volavg` | 未设 | 实数；Z_eff 无量纲 | parabolic 的指定体平均。 |
| `lineavg` | 未设 | 实数；Z_eff 无量纲 | parabolic 的指定算术节点平均，优先于 volavg。 |
| `alpha` | 1 | 有限实数 | parabolic 外幂，通常取正。 |
| `alpha1` | 2 | 有限实数 | parabolic 内幂，通常取正。 |
| `height` | 0 | 实数；Z_eff 无量纲 | gaussian 峰值。 |
| `integrated` | 未设 | 实数，体积积分单位 m³ | gaussian 非零时优先。 |
| `width` | 0.1 | 正有限实数，ρ 的单位 | gaussian 标准差。 |
| `center` | 0 | 实数，ρ 的单位 | gaussian 中心。 |
| `coefs` | [0,0,0,0] | 数字数组，使用前四项，不足补 0 | exponential 系数，公式同组分剖面。 |
| `profstr` | 未设 | 两行以分号隔开的数字字符串 | manual 的 ρ/ρ_edge 和正值数据，插值同组分。 |

## 运行中反馈 `[time_dependence]`

这是反馈控制，不是预先指定时变源/通量。启用时对阈值和倍率做额外检查。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `update_source` | `false` | 布尔 | 接受步后按 Q=P_fusion/P_aux 调整可控辅助热源幅值；外部后端沉积不随这个倍数乘；没有正的可控源时会跳过。不读取下一文件帧。 |
| `Q_threshold_low` | 2 | 非负有限实数，无量纲 | 低于阈值且仍有功率余量时增加辅助加热。 |
| `Q_threshold_high` | 10 | 有限实数 ≥ Q_threshold_low | 高于阈值时允许降低辅助加热，低高之间不调。 |
| `p_up_multiplier` | 1.01 | 有限实数 ≥ 1，无量纲 | 一次增加的幅值倍数。 |
| `p_down_multiplier` | 0.9 | 0 < 倍数 ≤ 1，无量纲 | 一次减少的幅值倍数。 |
| `p_aux_max_MW` | 100 | 有限实数 ≥ p_aux_min_MW，MW | 总辅助热功率上限；核算时包含外部沉积。 |
| `p_aux_min_MW` | 0.01 | 非负有限实数，MW | 总辅助热功率下限。 |
| `update_edge` | `false` | 布尔 | 允许按核心/边界比调高独立边界，保持组分链接和准中性约束；状态跳变重启 BDF 历史。 |
| `p_threshold_ratio` | 4 | 正有限实数，无量纲 | p_inner/p_edge 超过该值时允许增加 T_edge。 |
| `p_edge_multiplier` | 1.05 | 有限实数 ≥ 1，无量纲 | 尽管名称有 p，实际乘外侧温度，然后由 nT 重算 p。 |
| `T_edge_max` | 40 | 正有限实数，keV | 可调外侧温度上限。 |
| `n_threshold_ratio` | 4 | 正有限实数，无量纲 | n_inner/n_edge 超过该值时允许增加 n_edge。 |
| `n_edge_multiplier` | 1.05 | 有限实数 ≥ 1，无量纲 | 外侧密度调整倍数。 |
| `n_edge_max` | 40 | 正有限实数，10²⁰ m⁻³ | 可调外侧密度上限。 |

## 数据文件 `[import]`

同次运行共享一个 `[import]`，各组分的 `key`/`keys` 负责选场。ASCII 用空白分列，可用 `#` 注释；不是 CSV。TRANSP 的 `*2d` 同目录还要有同前缀同后缀的 `*0d`、`*1d`；TRANSP NetCDF 未实现。T3D `.npy` 使用 Python pickle，只应加载可信文件。此版本均选取固定径向快照，不沿时间轴切换。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `type` | 未设（不读） | `"columns"`（别名 `ascii/text/txt`）、`"trinity"`（`t3d`）、`"transp"`、`"jetto"` | 选择读取器。Import 通量仅接收 columns 或 trinity；TRANSP/JETTO 可提供状态/辅助源和支持的几何。 |
| `file` | 空字符串 | 路径 | ASCII 分列文本；T3D .log.npy/.nc/.bp；TRANSP ASCII U-file 的 *2d 文件；JETTO NetCDF。选择读取器后须给兼容文件。 |
| `columns` | [] | 字符串数组，名称唯一 | ASCII 逐列名称。状态/体源缺 rho 时会警告并假设均匀 [0,1]；通量必须给显式 rho。 |
| `divide_by` | []（各列视为 1） | 与 columns 等长的有限非零实数数组 | ASCII 各列先除以对应数，使输入达到下述物理单位。 |
| `frame` | −1 | 整数，Python 索引 | trinity 只选一帧；−1 为末帧，0 为首帧。通量要求所选帧诊断可用，坐标约定一致。 |
| `transp_time` | 0.0 | 时间戳数值，通常 s | TRANSP 的非零值选最接近时间片；建议显式指定非零时间，0 并不可靠地等同首帧。 |
| `jetto_time` | 未设（末帧） | 有限时间戳数值，通常 s | JETTO 选最接近时刻；可使用 0。 |
| `jett0_time` | 未设 | 兼容别名 | 写的是数字 0；仅 jetto_time 未指定时生效。 |

## 结果与检查点 `[log]`

`.log.npy` 是常规输出。命令行 `t3d case.in -l case.out`（或 `python -m t3d`）的 `-l/--log` 另设文本求解日志；省略为输入同干名 `.out`，不是 `[log]` 中的键。

| 参数 | 默认 | 类型、单位或可选值 | 含义与生效条件 |
|---|---|---|---|
| `f_save` | 输入文件不带 .in 的路径 | 输出文件干路径字符串（非目录） | 为 .log.npy、.nc、.bp 分别生成文件名；历史值带已知后缀时先剥离，避免覆盖。 |
| `output_netcdf` | `false` | 布尔 | 按 NetCDF4 分组输出，适合跨语言读写、核对单位与诊断有效性。 |
| `output_adios2` | `true` | 布尔 | 输出 ADIOS2 BP；需要可用的 ADIOS2 环境。关闭仅改变输出格式。 |
| `n_write` | 1 | 整数 ≥ 1 | 每 N 个已接受时间步写检查点，停止后仍写末帧；不代表只保存 Newton 迭代后的时间点。 |

## 通量模型 `[[model]]` 与外部源 `[[source]]`

### 模型选择与通用约定

以下参数由 `d5ac1fe` 直接读取；类型或范围写的是可用输入，不能视为已验证所有后端。省略表示 TOML 中不写该键，不是写入 `null`。外部程序自己的原生输入关键字不属于 T3D 参数表；模板可保留它们，T3D 为每次计算覆盖几何、组分、梯度和必需的运行/输出项。后端工作目录采用本次运行隔离，裸路径相对于启动目录。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `model` | 注册名或注明别名 | "GX" | 选择通量模型。见下表；全局可写 [model] 或多个 [[model]]。没有该表时默认尝试 GX。多个通量模型通常叠加；PowerBalance 例外。 |
| `label` | 非空字符串，不含 /；同一运行中各模型的输出标签唯一 | 各模型表所示 | 用于分项通量输出，例如 Q_ReLU_MW_H。模型和独立源项各自有自己的标签空间。 |

### ReLU — 参数化通量

粒子支使用 `[species.density_relu_flux]` 的零通量逆标长；热支使用 `[species.pressure_relu_flux]` 的**温度**逆标长阈值。阈值径向变化由组分表指定；模型级 `noise` 与组分表中的保留字段 `noise_frac` 不同。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `D_neo` | 有限实数；物理扩散应非负 | 0.0 | 经验背景项强度：粒子分支加 n κ_n D_neo，热分支加 p κ_p D_neo；不是自动 DKES/SFINCS 新经典系数。 |
| `power` | 有限实数 > 0 | 1 | 阈值后超额梯度的幂；可使用非整数，对反向梯度采用有符号幂。 |
| `noise` | 有限实数 [0,1] | 0 | 乘法噪声幅度；按物理步、组分、通量面、通道抽样。同一接受步的 Newton、扰动和重试共用该步抽样。 |
| `random_seed` | 非负整数 | 10 | 随机种子。 |
| `label` | 输出标签 | "ReLU" | 模型输出名称；模型名也接受 relu。 |

### ChangHinton — 解析新经典离子热通量

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `label` | 输出标签 | "Chang-Hinton" | 只支持 [geometry].geo_option = "miller"。用第一种离子及局域磁面参数估算热输运；无粒子通量。不另读扩散系数文件。 |

### RR — Rechester–Rosenbluth 型电子热通量补项

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `alpha_rr` | 有限实数 ≥ 0 | 1.0 | 局部热通量系数；电子热通量与温度逆标长 κ_T,e 的三次方成正比，0 关闭该项；非零时需有限正的电子碰撞性。 |
| `label` | 输出标签 | "RR" | 模型输出名称，仅电子能量通道有此补项。 |

### PowerBalance — 功率与粒子平衡通量

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `label` | 输出标签 | "PowerBalance" | 用源项在活动壳层中的积分反推穿面通量；是平衡诊断闭合而非独立预测输运，不能与其他通量模型或 turbulent_exchange 并用。 |

### Import — 文件给定穿面通量

接受 ASCII 空白分列文本或 T3D 快照中的物理诊断；不读任意 TRANSP/JETTO 通量或随时间推进的序列。通量面输入是整个磁面的有符号粒子率（10²⁰ s⁻¹）或热功率（MW）；不是热流密度，也不是已经归一化的 gyro-Bohm 通量。不是源项，不贡献剖面 Jacobian。单独一个数据列可按组分标记回落到 `Gam_SI20_<tag>` 或 `Q_MW_<tag>`。文件中不得有非零轴上穿面量，禁止径向外推；在读入时选定一帧后固定。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `label` | 输出标签 | "Import" | 模型每次运行只能注册一次；模型名也接受 import。配置 [import] 和相应组分的 particle_flux、heat_flux；必须至少启用一个活跃通道。 |

### GX — 陀螺动理学通量

当前求解器在每个活动面及场线位置生成原生 GX TOML，读回 `.out.nc` / `.nc`，依据诊断数据结构兼容两种模式。平衡几何可由 VMEC、DESC、GEQDSK、Miller 生成。局部资源配置受 `GK_SYSTEM`（默认 `local`；内置支持 stellar、traverse、satori、perlmutter、polaris、summit）、`CUDA_VISIBLE_DEVICES`（无时尝试 `nvidia-smi`）和 Polaris 的 `PBS_NODEFILE` 影响。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `gx_path` | 含可执行 gx 的目录 | 环境 GX_PATH，否则空目录前缀 | 显式 TOML 键优先。若未 dry_run，执行文件必须可执行。 |
| `gx_template` | TOML 文件路径 | "tests/regression/gx_template.in" | 物理及数值模板。保留分辨率、边界等模板设置；T3D 填入局域几何、组分和梯度，并追加诊断/重启项。 |
| `gx_outputs` | 目录路径 | "gx/" | 外部输出根；每个运行建隔离子目录，不复用旧运行的任意文件。 |
| `gx_logfile` | 文件名 | "gx.log" | 该后端日志，另有各次调用日志。 |
| `overwrite` | 布尔值 | false | 兼容旧输入。此提交读取但不用于放开跨运行输出复用。 |
| `gpus_per_gx` | 正整数 | 1 | 每个通量管使用的 GPU 数；Polaris 内置启动器上限为 4。 |
| `gx_max_concurrent` | 正整数或省略 | 省略 | 同一时间并行的 GX case 上限；本地/Polaris 还受分到的 GPU 组限制，自定义/其余集群启动器省略时为 1。 |
| `run_command` | 字符串（按 argv 解析）或字符串数组 | 省略 | 自定义启动命令前缀，如 "srun ..."；不交给 shell，不解释管道或重定向。 |
| `electromagnetic` | 布尔值 | true | 使用局域参考 β；false 写入很小的 5×10⁻⁴ 来近似静电极限。 |
| `zeta_center` | 有限标量或非空列表（弧度） | [0.0] | 仿星器通量管中心；多个位置用于场线平均或曲面插值。 |
| `theta_zeta_fluxes` | 布尔值 | false | 用有权重的周期 θ-ζ 曲面平均替代场线的算术平均；需多个不同位置并能覆盖曲面。 |
| `adiabatic_species` | 已声明组分的 type 字符串或省略 | 省略 | 指定由 GX 作为 Boltzmann 组分处理的粒子。species 中的 adiabatic 声明优先；参考组分必须为动力学组分，且至少有一个动力学组分。 |
| `collisionality_scaling_factor` | 有限实数（物理上非负） | 1.0 | 发送给 GX 的碰撞频率乘子。 |
| `qflux_scaling_factor` | 有限实数 | 1.0 | 返回 GX 热通量乘子，用于灵敏度测试。 |
| `tprim_shift` | 有限实数（无量纲） | 0.0 | 加到 GX 的温度逆标长 tprim 的常量偏移。 |
| `no_perturb_temperature` | 组分 type 的字符串数组 | [] | 这些独立温度不做 GX 有限差分扰动，例如某些 ETG 设置；不等于禁用 T3D 温度演化。 |
| `dkap_T` | 有限正实数（无量纲） | 0.5 | 计算 GX 温度梯度响应所用的绝对扰动下限。 |
| `dkap_n` | 有限正实数（无量纲） | 0.5 | 计算 GX 密度梯度响应所用的绝对扰动下限。 |
| `check_stalls` | 布尔值 | false | 监视子进程输出是否停止增长；资源排队时宜谨慎使用。 |
| `monitor_time` | 正实数（s，启用监视时） | 10 | 监视周期。 |
| `stall_abort_count` | 正整数（启用监视时） | 6 | 连续无增长检查达到此次数后中止。 |
| `build_ReLU` | 布尔值 | false | 使用 GX 结果估计各面代理模型，用于后续步；重试和几何变化可使代理失效。 |
| `nsteps_ReLU` | 正整数（启用代理时） | 1000 | 重新调用 GX 的物理步间隔，不是 Newton 迭代次数。 |
| `abort_on_nans` | 布尔值 | true | 检查最终汇总的粒子、热与交换通量是否全有限；关闭不会取消文件读取阶段的有限性检查。 |
| `no_restarts` | 布尔值 | false | true 禁用 GX 热启动。默认也只允许同几何且允许重启的同次运行数据，不等于从旧工作目录回放。 |
| `dry_run` | 布尔值 | false | 跳过 GX 可执行文件检查/启动，以产生 GX 输入。没有相应输出时不能完成有效的输运回算，不是无 GX 的求解模式。 |
| `chie_over_chii` | 有限实数（物理上非负） | 0 | 非零时以给定 χₑ/χᵢ 从离子热通量推定电子热通量；仅限绝热电子与一个动力学离子。 |
| `label` | 输出标签 | "GX" | 模型输出名称；模型名也接受 gx。 |

### GS2-QL — 线性陀螺动理学谱与准线性通量

面几何传递分支为 GEQDSK 或 Miller；模板用 Fortran namelist，不是 GX 的 TOML。单 case 的 `.out.nc` 提供线性和几何诊断；不足两个不同实际 (k_y) 的谱不允许积分。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `gs2_path` | 含 gs2 或 multigs2 的目录 | 环境 GS2_PATH，否则空目录前缀 | 显式 TOML 键优先；目标必须可执行。 |
| `mpiexec` | mpirun、srun、multigs2、mpirun-multigs2 | "mpirun" | 分别使用逐 case MPI/Slurm 启动或批量 multigs2；批量模式要求配套可执行程序。 |
| `gs2_template` | Fortran namelist 文件路径 | "tests/unit/gs2_template.in" | 线性 GS2 配置模板；T3D 写入几何、模态与组分并填必要输出开关。 |
| `gs2_outputs` | 目录路径 | "gs2/" | 外部作业输出根，每次 T3D 运行建独立子目录。输出 .out.nc 包括增长率、谱、通量响应等供准线性回算。 |
| `gs2_logfile` | 文件名 | "gs2.log" | 该后端日志。 |
| `overwrite` | 布尔值 | false | 传给 GS2 输入 namelist 写出函数的 force；重名 trial 仍拒绝，不表示允许重用旧结果。 |
| `cpus_per_gs2` | 正整数 | 1 | 每个 GS2 case 的 MPI 进程数；不超过 nproc。 |
| `nproc` | 满足 nproc ≥ cpus_per_gs2 的正整数 | 1 | 分配总进程数；并行 case 数为其与每 case 进程数比值的整数部分。 |
| `electromagnetic` | 布尔值 | true | 是否走电磁设置；false 关闭电磁势相关项。 |
| `gammaE` | 有限标量或长度 N_radial−1 的列表 | 0 | 各面归一化 E×B 剪切率；非零时气球角扫描必须覆盖计算所需区间。 |
| `bakdif_on_template` | 布尔值 | false | true 从原模板保留相关分布函数扩散设置；false 写入 fexpr=0.48、bakdif=0.05。 |
| `nky` | 整数 ≥ 2 | 2 | 默认 k_y 谱采样数；use_toroidal_n 时最终取 nn 的长度。 |
| `kymin` | 有限实数 | 0.1 | 归一化 k_y 扫描起点；谱积分需要至少两个不同的实际 k_y。 |
| `kymax` | 有限实数 | 1.0 | 归一化 k_y 扫描终点；建议大于起点。 |
| `theta0` | 有限角度数组（弧度） | [] | 非空则覆盖 ntheta0/th0min/th0max；多点数组必须从 0 严格递增。 |
| `ntheta0` | 正整数 | 1 | theta0 为空时使用的气球角采样数。 |
| `th0min` | 有限实数（弧度） | 0 | 气球角起点；多点要求从 0 开始。 |
| `th0max` | 有限实数（弧度） | 0 | 气球角终点；多点要求递增，并覆盖所需剪切积分区间。 |
| `use_toroidal_n` | 布尔值 | false | true 使用环向整数模数 nn 而非 k_y 均匀扫描。 |
| `nn` | 至少两项的环向模数列表 | 启用 use_toroidal_n 时必填 | 选定环向模，最终必须形成至少两个不同的实际 k_y。 |
| `label` | 输出标签 | "GS2" | 模型选择名为 GS2-QL 或 gs2-ql；热速度归一化为 √(2T/m)，不可与声明相反约定的模型混用。 |

### SFINCS — 局域新经典通量

支持 `GK_SYSTEM` 为 local（默认）、perlmutter、stellar、polaris、summit；Polaris 的内置启动需 `PBS_NODEFILE`。其他环境回退到本地 MPI 启动，可用 `run_command` 覆盖。真实运行验证范围仅为既有 W7‑X CPU 冒烟，不等同于完整时序和并行扩展验证。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `sfincs_path` | 含 sfincs 的目录 | 工作目录 | 非空环境 SFINCS_PATH 优先于此键。 |
| `sfincs_template` | Fortran namelist 路径 | "tests/regression/sfincs_template.namelist" | T3D 填好 VMEC 几何文件、组分、rho、密度/温度导数；支持 VMEC 或以 VMEC 格式导出的 DESC，要求 torflux。 |
| `sfincs_outputs` | 目录路径 | "sfincs/" | 作业输出根；每次 T3D 运行建隔离目录，按试次/面生成 input.namelist 和 sfincsOutput.h5。 |
| `sfincs_logfile` | 文件名 | "sfincs.log" | 该后端日志。 |
| `overwrite` | 布尔值 | false | 兼容旧输入。该提交读取但不用于跨运行复用。 |
| `run_command` | 字符串（按 argv 解析）或字符串数组 | 省略 | 自定义运行命令前缀；应按实际任务数分配资源。 |
| `cpus_per_sfincs` | 正整数 | 1 | 每 MPI 进程的线程数；通过 OMP_NUM_THREADS 传递。 |
| `mpis_per_sfincs` | 正整数 | 1 | 每个磁面 case 的 MPI 进程数。 |
| `cpus_total` | 正整数；至少覆盖一个 case | 可见 CPU 数（优先进程 affinity） | 最大并发 case 按 cpus_total / (cpus_per_sfincs × mpis_per_sfincs) 向下取整；Polaris 另受实分节点约束。 |
| `label` | 输出标签 | "SFINCS" | 模型选择名也接受 sfincs；SFINCS 输出换算后按 T3D √(T/m) 约定，可与 GX 相加。 |

### KNOSOS — 仿星器新经典通量

由 `booz_xform` 预处理 wout 得到 `boozmn.nc`；`GK_SYSTEM` 的内置启动分支是 local（默认）、perlmutter、stellar、summit。其余环境需显式 `run_command`。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `knosos_path` | 含 knosos.x 的目录 | 工作目录 | 非空环境 KNOSOS_PATH 优先于此键。 |
| `knosos_outputs` | 目录路径 | "knosos/" | 作业输出根。T3D 生成 input.model、input.species、input.surfaces、input.parameters 和 profiles_trin.txt，读每个面的 knosos_for_predictive.txt.NN。 |
| `knosos_logfile` | 文件名 | "knosos.log" | 该后端日志。 |
| `adiabatic_electrons` | 布尔值 | false | true 在输入中用绝热电子设定。不可与 T3D 独立电子输运状态混淆。 |
| `overwrite` | 布尔值 | false | 兼容旧输入。该提交读取但不用于跨运行复用。 |
| `run_command` | 字符串（按 argv 解析）或字符串数组 | 省略 | 自定义命令前缀。默认一个计算需 N_radial−1 个 CPU 任务。 |
| `cpus_total` | 正整数且 ≥ N_radial−1 | 本地可见 CPU 数；已知集群默认 N_radial−1 | 按 CPU 容量限定并发的 KNOSOS case 数。 |
| `DIRDB` | 路径字符串 | 省略 | 可选数据库路径，写入 input.surfaces；语义由 KNOSOS 决定。 |
| `DIRS` | KNOSOS namelist 可解析的值（原样行文本） | 省略 | 传给 input.surfaces 的后端选项；范围以所用 KNOSOS 版本为准。 |
| `SDKES` | KNOSOS namelist 可解析的值（原样行文本） | 省略 | 传给 input.surfaces 的 DKES 相关后端选项；范围以所用 KNOSOS 版本为准。 |
| `label` | 输出标签 | "KNOSOS" | 模型选择名也接受 knosos。需 booz_xform 和可预处理的 VMEC 格式几何；内部 KNOSOS 面坐标为 s=ρ²。 |

### NEO — GACODE 轴对称新经典通量

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `neo_template` | KEY=VALUE 文本路径 | 必填 | 需 N_RADIAL=1、PROFILE_MODEL=1、SILENT_FLAG=0（可由接口补入）；SIM_MODEL 若显式指定为 1 或 2。使用 Miller 类几何的局域参数。 |
| `neo_outputs` | 目录路径 | "neo" | 作业输出根。每个磁面/扰动生成 input.neo，读 out.neo.transport。 |
| `qmax` | 有限实数；≤0 关闭 | 0 | 正值仅裁剪最内传输面热通量的绝对值（模型归一化）；保留方向。 |
| `mpiexec` | mpirun、srun、local、none、空串 | "mpirun" | NEO 每个 case 一个 MPI rank；local/none/空串不加启动前缀。 |
| `cpus_per_task` | 正整数 | 1 | 每 case 的 CPU 数，也是 OMP_NUM_THREADS。 |
| `nproc` | 正整数且 ≥ cpus_per_task | 可见 CPU 数 | 总 CPU 预算。 |
| `mem_per_cpu` | 正数（GB/CPU） | 5 | 仅 srun 时传入 --mem-per-cpu。 |
| `label` | 输出标签 | "NEO" | 模型选择名也接受 neo。GACODE_ROOT 必须指向包含 neo/bin/neo_parse.py 与 neo/src/neo 的安装；接口声明 √(2T/m) 约定。 |

### BEAMS3D — 外部中性束源项

几何需 `torflux` 且有 VMEC 格式 `wout*.nc`；不能只改变 T3D 的尺寸或磁场参考值而让外部程序继续使用原始磁平衡，否则会报错。HDF5 后处理按束求和，再插值到状态剖面节点，而不是求输运通量。支持接口/合成数据验证，不宣称完成真实 BEAMS3D 工况验证。

| 参数 | 类型／可选值 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `source` | BEAMS3D（区分大小写） | 设置 [source] 时必填 | 目前唯一注册的外部源；可用 [source] 单表或 [[source]] 数组。 |
| `label` | 唯一源项标签 | "BEAMS3D" | 输出分项名，多个束模型需不同标签；不能用 aux、alpha、rad、coll、heating。 |
| `particle_source` | 布尔值 | false | 是否加入热化后粒子源；启用时有功率的束同位素需匹配主离子（质量匹配容差 2%）。离子源与对应电荷权重的电子源同时构造。 |
| `power_source` | 布尔值 | true | 是否加入电子和主离子慢化能量沉积；与 particle_source 不得同时为 false。 |
| `beams3d_path` | 含可执行 xbeams3d 的目录 | 环境 BEAMS3D_PATH 或 "." | 显式 TOML 键优先。 |
| `beams3d_template` | 含 &BEAMS3D_INPUT 的 Fortran namelist 路径 | 必填 | 配置束能量、质量、功率、几何等本机参数；T3D 写入背景电子/离子温度（eV）、密度（m⁻³）、离子质量（kg）和 s=ρ²。 |
| `beams3d_outputs` | 目录路径 | "beams3d" | 输出根。每次驾驶器和每次试算都用独立目录；从 beams3d_t3d.h5 读取径向网格、束粒子和电子/离子功率沉积。 |
| `beams3d_logfile` | 文件名 | "beams3d.log" | 每个真实作业的标准输出/错误日志。 |
| `run_command` | 字符串（按 argv 解析） | 省略 | 启动器前缀；省略时直接执行。 |
| `run_options` | 字符串（按 argv 解析） | "" | 追加到 -vmec t3d -collisions 的本机参数，不得用 -depo、-hitonly、-hit_only、-plasma。 |
| `implicit` | 布尔值 | false | true 时对独立 n,p 扰动重算沉积，形成外部源项 Jacobian；alpha=0 或无演化剖面时不执行扰动。 |
| `relative_step` | 有限实数 (0,1) | 0.001 | 隐式源项 Jacobian 的相对剖面扰动步长。 |
