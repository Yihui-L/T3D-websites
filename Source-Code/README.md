# T3D-lyh 源码手册

TRINITY3D · 一维径向输运

原版 T3D 对应上游 v0.1.0 提交 `74a06df`。本仓库的 `Source-Code/T3D-lyh/` 保存维护版源码。本页按模块说明物理模型、方程、I/O 和相对原版的累计修改。[原版源码](https://bitbucket.org/gyrokinetics/t3d) · [T3D-lyh 源码](https://github.com/Yihui-L/T3D-websites/tree/main/Source-Code/T3D-lyh)

维护此页时，将同一功能或影响范围的修改并入已有条目，独立主题另列一项。条目随源码更新，说明当前行为、适用范围及相关验证。

[源码版本说明](SOURCE_VERSION.md) · [在线程序手册](https://yihui-l.github.io/T3D-websites/Source-Code/) · [完整输入参数](PARAMETERS.md) · [W7-X ReLU 算例](https://yihui-l.github.io/T3D-websites/W7X-ReLU-lyh/)

## 章节目录

- [01 · 输运方程与数值求解](#transport)
- [02 · 平衡与几何](#geometry)
- [03 · 物理源项与反馈](#physics)
- [04 · 通量模型与并行调度](#closures)
- [05 · 输入、重启与诊断输出](#io)

<a name="transport"></a>

## 01 · 输运方程与数值求解

### 控制方程与离散

令 ρ 为径向磁通面标签，V(ρ) 为包围体积，V′=dV/dρ。组分 s 的未知量是粒子密度 nₛ 和热压 pₛ=nₛTₛ（T 按能量计）。用穿过整张磁面的向外粒子率 Γₛ 和热功率 Qₛ 写磁面平均方程：

```text
∂nₛ/∂t        = Sₙ,ₛ − (1/V′) ∂Γₛ/∂ρ
(3/2) ∂pₛ/∂t  = Sᴱ,ₛ − (1/V′) ∂Qₛ/∂ρ
Γₛ = V′⟨Γ⃗ₛ·∇ρ⟩,   Qₛ = V′⟨q⃗ₛ·∇ρ⟩
ΔVⱼ dUⱼ/dt  = Fⱼ₋½ − Fⱼ₊½ + ΔVⱼ Sⱼ
```

前两式中的源分别是粒子率密度和体功率密度；最后一式同时适用于 U=n 和 U=3p/2。在程序中活动壳的 ΔVⱼ 按节点几何权重离散（约为 V′ⱼΔρ）。两个壳共享同一界面通量，因此相加时内部通量精确抵消；轴面通量为零，外侧剖面节点固定，最后一个活动壳流出面是相邻节点的中点。电荷约束是 Σₛ Zₛnₛ=0；从属密度由约束重建，温度通过 pₛ/nₛ 得到。默认不推进电流、快离子分布或三维平衡。

**时间离散。** 令 D(y)=源−通量散度。对 θ（输入 `[time].alpha`）小于 1，求 (yₙ₊₁−yₙ)/h = θD(yₙ₊₁)+(1−θ)D(yₙ)。θ=0 为显式 Euler；θ=1 首步/历史重置后用后向 Euler，之后用变步长 BDF2。若 r=h/h₋，BDF2 导数为

```text
[(1+2r)/(1+r) · yₙ₊₁ − (1+r) · yₙ + r²/(1+r) · yₙ₋₁] / h
```

隐式迭代对同一离散残差线性化；Newton 更新量、时间离散残差和空间稳态残差分别记录。只有接受的步进入历史，拒步恢复旧态。默认 Newton 接受容差不等于实验或网格收敛。论文参见 [Barnes 等：TRINITY](https://doi.org/10.1063/1.3323082)、[Giacomin 等：T3D 方程与准线性闭合，§4.1](https://doi.org/10.1017/S0022377824001107)。

**I/O。** `[grid]` 决定节点和面，`[time]` 控制时间离散，`[[species]]` 设初值、边界、演化及约束；求解器读取界面 Γ/Q 与节点源项，输出 n、p、T、时间指标和功率余额。详见[参数表](PARAMETERS.md)。

### 相对上游的改动

- **变步长 BDF2：** 时间系数只使用已接受的历史步长；拒步恢复剖面并重算 θ 法所需的旧时刻输运项，不把试探状态写入历史。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Time.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_lyh_time_control.py)
- **时间步控制：** 校验时间参数与重试上限；首步和增步遵守 dtau_max，SI 输入只转换有量纲的时间量，不转换缩步比或误差阈值；到达终止时刻时消除仅由舍入产生的额外小步。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Time.py)
- **Newton 求解：** 以差分形式构造时间离散残差并直接求解修正量，降低大项相减误差；Jacobian 用同一离散残差、受约束剖面和后端梯度响应，包含内部碰撞及热源导数。显式推进跳过 Jacobian。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Solver.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_discrete_jacobian.py)
- **收敛诊断：** 分开记录 Newton 更新量、时间离散残差、半离散稳态速率；稳态停止看实际输运速率，不因某次 Newton 更新很小而停。时间离散残差诊断不改变默认 Newton 接受阈值。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Solver.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_numerics_round3.py)
- **共享界面与磁轴：** 相邻控制体共用同一个物理粒子／热通量；磁轴通量严格为零，指定外侧剖面节点不参与推进或活动壳积分。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_vmec_source_balance.py)
- **离散边界与终止：** 固定边界作为代数约束而非 BDF 外推；遇 stop 文件回滚未接受试探步；零时长或无演化变量时只写诊断，不强行推进。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/trinity_lib.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_driver_transactions.py)
- **径向网格与紧致梯度：** 修复三点网格的节点／界面插值和边界导数。紧致差分用等价带状求解；保留浮点精度并拒绝非法网格、剖面和零差分步。未测量加速比。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Profiles.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_numerics_round3.py)
- **准中性与组分约束：** 一致处理从属物种、关联密度和温度剖面；支持显式选定准中性从属组分，拒绝重复组分、冲突约束和非正／非有限的密度或压强试探态。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_species.py)
- **磁面准中性：** 密度插值发生正值回退时，各物种改用同一线性模板；同步约束基础和扰动的电荷梯度，输运模型、Jacobian 和输出使用同一套面梯度。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_numeric_constraints.py)
- **剖面算术与 RMS：** 剖面持有自己的数据，拒绝同长度但不同坐标的混算；Newton 的缩放 RMS 在极大、极小和空状态下不会意外溢出、下溢或崩溃。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Profiles.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_numeric_constraints.py)

<a name="geometry"></a>

## 02 · 平衡与几何

### 磁面坐标、几何与工作范围

仿星器采用 `torflux`，即 ρ=√s、s=Φₜ/Φₜ,LCFS；有 dV/dρ=2ρ·dV/ds。若 A 是磁面面积，几何关系为 A=V′⟨|∇ρ|⟩；只在得到正确磁面投影或面积平均后，才可把热流密度换成穿面功率。代码保存 A/a²、a⟨|∇ρ|⟩ 等无量纲度量，活动壳近似体积是 a³Δρ·(A/a²)/(a⟨|∇ρ|⟩)。`rho_edge` 是指定的剖面边界，不必是 LCFS。

**I/O。** `[geometry]`/`[rescale]` 选择平衡和尺寸：`vmec` 读 `wout*.nc`（半网格 Nyquist (0,0) 雅可比提供体积导数；本版拒绝不支持的非对称文件）；`desc` 读原生平衡文件（可按后端需要导出 VMEC）；`geqdsk` 在 `rminor` 网格上由唯一兼容的 GX 或 GS2-QL 提供几何；`miller` 用兼容导入取得参数；`basic` 是解析简化几何。输出含节点/面面积、径向度量和逐帧几何记录。默认固定几何；在线 DESC 更新仅支持保持演化壳体积与参考小半径的收敛候选，不含移动磁面、压缩功或自举电流闭合。不同文件与后端的条件见[参数表](PARAMETERS.md)。

### 相对上游的改动

- **VMEC 体积导数：** 从半网格 Nyquist (0,0) 雅可比系数取 dV/ds，再按 ρ=√s 计算 dV/dρ；径向节点和输运面共用一致的面积／体积度量。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/geometry_models/VMEC.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_vmec_source_balance.py)
- **磁面面积：** 周期角度采样不重复积分接缝；遇到不支持的非对称 wout 明确报错，不静默忽略非对称项。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/geometry_models/VMEC.py)
- **基本几何与重标度：** 圆形基本几何的 A/a² 和 a|∇ρ| 使用一致无量纲定义，修正磁场默认与 VMEC LCFS 回退；径向线平均不再带长度单位，非法度量明确报错。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Geometry.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_driver_transactions.py)
- **GEQDSK／Miller：** 由实际能力选取唯一 GX 或 GS2-QL 几何提供者，不依赖自定义标签；GEQDSK 用 rminor 网格，逐面读入几何。Miller 必须启用兼容的导入，Boozer 预处理按后端能力执行。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Geometry.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_gs2_ql_geometry_and_metric.py)
- **DESC 平衡反馈：** 在导入 JAX 后端前选定 CPU／GPU；压强增减均可触发，分开保存候选平衡，仅接受收敛结果；按需要输出 VMEC 格式。当前只接受不改变演化壳层体积／参考小半径的更新；没有移动磁面输运或压缩功。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/geometry_models/DESC.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_geometry_balance_interfaces.py)
- **DESC 体积权重与 VMEC 输入：** DESC 直接插值原生 dV/dρ，再用面积一致地构造度量，明确要求 ρ=√s。VMEC 拒绝掩蔽输入并确保读取失败时关闭 NetCDF。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Geometry.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_iterative_io.py)
- **几何积分与逐帧输出：** 修复节点和输运面的插值体积积分，逐帧保存度量；功率分解按所选帧取几何。Miller 重启只导入物理几何，不继承旧运行的模型和目录。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Geometry.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_iterative_io.py)

<a name="physics"></a>

## 03 · 物理源项与反馈

### 源与汇的物理模型

离子升温、电子降温可以来自同一个成对换热项。以 T 为能量计，
`Sᴱ,coll,ₛ = (3/2) nₛ Σᵤ νᴱₛᵤ (Tᵤ−Tₛ)`。物理组分全部演化时，成对交换总和应为零；固定某个温度时，它相当于规定热库，不应声称该简化系统独立闭合总热量。径向热流不是碰撞体热源。

- 辅助粒子源与辅助能量源可规定高斯形状 `A·exp[−(ρ−ρ₀)²/(2σ²)]`。若指定 `integrated`，用活动壳的 Σⱼ ΔVⱼSⱼ 反求幅值；否则使用 `height`。也可导入离线沉积或使用 BEAMS3D 在线计算。
- D–T α 出生功率密度是 `Eα nᴰ nᵀ ⟨σv⟩(Tᵣₑₗ)`，Eα≈3.5 MeV；两 Maxwellian 温度独立时，`Tᵣₑₗ=(mₜTᴰ+mᴅTᵀ)/(mᴅ+mₜ)`。沉积功率等于出生功率乘 `alpha_efficiency`（且需开启热反馈），再按电子/离子份额分配；聚变总出生功率为 α 出生功率约 5.03 倍。该通道不等同于求解燃耗、灰积累或快粒子输运。
- 杂质辐射是电子能量汇，按 `nₑΣᵢnᵢLᵢ(Tₑ)` 组装；包含谱线、复合和轫致辐射的模型开启时，不重复加独立的轫致辐射 `∝ Zₑff nₑ²√Tₑ`。可另计通量后端提供的湍流组分间能量交换。
- 可选功率或边界反馈在已接受步之间更新；Sudo 是经验密度控制，不是额外演化方程。辅助功率反馈也不读取时变文件。

**I/O。** `[physics]` 开关物理源，`[[species]].aux_particle_source / aux_power_source` 给规定源，`[[source]]` 选择外部沉积后端，`[time_dependence]` 控制反馈。正的热源表示向该组分加热，诊断以分项和总量分别输出；活动壳功率积分排除固定边界节点。完整[源参数和配置](PARAMETERS.md)。

### 相对上游的改动

- **高斯源项归一化：** 对活动壳做体积积分；显式给定总源（包括零）按活动域归一化，不把固定外节点当作沉积单元；对域外中心的截断尾部给出警告，不把几乎无重叠误当作正常输入。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_vmec_source_balance.py)
- **D–T 反应与 α 沉积：** 区分 α 出生功率、沉积热和聚变增益。关闭 α 热反馈或降低保留率不改变出生功率；独立 D/T 用两个 Maxwell 分布的相对温度。纯 H/e 不产生 D–T 功率。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Physics.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_physical_recurrence.py)
- **碰撞／辐射：** 用碰撞成对能量交换校验输运方程与源项导数；辐射模型已含轫致辐射时避免重复计入；关闭源项时清除残留诊断。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Physics.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_source_physics_io.py)
- **Zeff 与手动剖面：** 修复高斯／手动 Zeff 输入和混合电荷背景的杂质稀释，保持准中性；手工 n、T 表兼容 NumPy 2.x 并检查输入。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_physical_recurrence.py)
- **辅助功率控制：** Q 控制使用出生聚变功率，把外部沉积计入增益分母，但仅调节可控的规定辅助功率；没有可调热源时不虚构反馈。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_source_physics_io.py)
- **Sudo 与外侧边界：** 在初始化及已接受步之间应用密度反馈，单次通量、源项和残差计算使用同一个剖面。边界控制保持固定量、关联温度和准中性；发生跳变后下一步从一阶重启。Sudo 是经验反馈，不是额外演化方程。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/trinity_lib.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_driver_transactions.py)
- **PowerBalance：** 独立的源平衡通量闭合在所有源项收集后计算，不能作为额外通量叠加到其他闭合。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/PowerBalance.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_geometry_balance_interfaces.py)
- **BEAMS3D：** 改用完整碰撞追踪；输入按 s、eV、m⁻³、kg 写入，输出用 √s 径向箱并汇总束线。修正 Fortran 起始下标、稀疏束线与同位素检查，分别控制粒子和能量源；可选独立 n,p 差分导数需要低标记噪声。目前离子沉积映射到主离子。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/source_models/BEAMS3D.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_beams3d_model.py)
- **剖面依赖与中性根：** 按依赖顺序处理 init_to / equal_to；若约束本身已使电荷恒为零，保留独立演化根；若单一密度根的比例与准中性冲突，给出明确错误。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Species.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_physics_round4.py)
- **初始化与源项防护：** 自定义剖面组合平均值和边界值时保持固定边界；拒绝无物理意义的 Zeff 和非有限源项，清理外部源标签并校验 Q 控制范围。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Physics.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_physics_round4.py)
- **BEAMS3D 一致性：** 不把仅在 T3D 中重标度的几何默认为真实束注入几何；新调用清除上一份输出，无独立未知量时不构造空 Jacobian。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/source_models/BEAMS3D.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_beams3d_model.py)

<a name="closures"></a>

## 04 · 通量模型与并行调度

### 通量闭合、单位与后端

定义 κₙ=−∂ρln n、κₚ=−∂ρln p，故 κₜ=κₚ−κₙ。ReLU 热支路（关闭噪声时）采用 `m·sgn(κₜ)·max(|κₜ|−κc,0)^b`，κc 可以随 ρ 线性变化；粒子支路使用 `sgn(κₙ−κc)·|m(κₙ−κc)|^b`，不是热支路的阈下关断。可另加 `D_neo nκₙ` 和 `D_neo pκₚ`。以上是归一化闭合，不是直接输入 SI 扩散系数，亦不区分 ITG/TEM 等湍流模态。换成穿面 Γ/Q 时还要乘面几何和参考组分的局域 gyro-Bohm 因子。

- **内置计算：** ReLU 是参数化闭合；Chang–Hinton 是解析新经典近似，需适用几何；RR 是仅补到电子热流的 κₜ³ 型模型；PowerBalance 由累计源反推平衡通量，独占使用，不是预测模型。
- **外部计算：** GX 与 GS2-QL 提供湍流/准线性输运；SFINCS、KNOSOS 与 NEO 提供新经典输运。T3D 给出各面当前态及梯度，按后端需要提交扰动计算；后端的局部响应进入隐式迭代。模板和可执行程序不是外部通量表；并行与缓存只改变调度方式，不应改变方程。
- **文件给定：** `model = "Import"` 读静态穿面 Γ（10²⁰ s⁻¹）或 Q（MW），用有符号线性插值到活动通量面，不外推；保持维度值，不给 Jacobian 增添虚假剖面导数。允许与非重复的在线分量相加，不应在回放总量后又叠加已含的模型。

**I/O。** `[[model]]` 可重复声明兼容的加性闭合，用唯一标签区分输出；特定后端依赖其输入模板、所需平衡和运行环境。末帧不同步的外部诊断不标为有效。计算能力与实跑验证是两件事；见[模型与外部源完整参数](PARAMETERS.md)和[逐项改动](#closures)。

### 相对上游的改动

- **ReLU：** 修复反向梯度的有符号通量；开启噪声时，同一步的基值与扰动共享抽样，不依赖全局随机种子，避免噪声直接污染有限差分。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/ReLU.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_numerics_round3.py)
- **Rechester–Rosenbluth：** 在输运面使用同位置的参考归一化，补齐固定压强梯度下的密度扰动响应，不改原三次依赖。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/RR.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_numerics_round3.py)
- **GX 闭合：** ReLU 代理保留有符号的湍流能量交换 H 及梯度；先用完整物种集构造受约束梯度扰动，再筛选动力学组分。修复逐半径／角度的 GEQDSK 读取，并检查单样本与非有限输出。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/GX.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_backend_state_and_diagnostics.py)
- **GS2-QL：** 修复 GEQDSK 初始化和逐面读取；修正纯静电、零流剪切极限、单 k_y 取样以及有符号的组分权重。准线性模型见 Giacomin 等（2025）。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/GS2_QL.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_gs2_ql_geometry_and_metric.py)
- **SFINCS：** 平衡文件拷贝完成后才启动求解；控制 MPI 秩与线程预算，检查 HDF5 完成标志、非线性收敛、形状及有限值，不把半写或失败输出当作有效通量。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/SFINCS.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_backend_state_and_diagnostics.py)
- **GS2／NEO／KNOSOS 作业：** 用有界 CPU 调度和退出状态代替背景 shell 猜测；NEO 的输入解析器成功后才求解，KNOSOS 的径向输出必须齐全。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/_external.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_external_transport_jobs.py)
- **GX GPU 与 MPI 调度：** 本机按设备分组、每个任务按 GPU 数申请 MPI 秩；修正 Polaris 三卡分配，支持 Slurm／自定义启动器限流和可选的无进展报错。调度测试未用真实 GPU 或 Slurm。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/GX.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_external_transport_jobs.py)
- **拒步、缓存与失败清理：** 每次运行使用独立目录，步／迭代／拒步重试有独立标识；检查子进程退出和全部预期输出，终止仍在运行的兄弟作业。GX 仅把已成功并与同一几何匹配的检查点作热启动，不复用旧通量。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/trinity_lib.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_external_transport_jobs.py)
- **GX 模板与 SFINCS 归一化：** GX 以 TOML 解析含引号、缩进和标点的模板；SFINCS 保留物理电荷符号，并与输运面使用的参考速度和通量单位一致。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/SFINCS.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_external_flux_formats.py)
- **NEO 与 KNOSOS 通量读回：** NEO 核对物种、径向输出形状和归一化；KNOSOS 按预测输出约定读取主离子、电子及杂质通量，检查半径和列数。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/KNOSOS.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_external_flux_formats.py)
- **GX θ–ζ 图：** 拒绝掩蔽或非有限样本；仅对采样覆盖的区域插值，不把未覆盖区域推断为微小非零通量。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/GX.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_external_flux_formats.py)

<a name="io"></a>

## 05 · 输入、重启与诊断输出

### 输入 → 计算 → 输出

1. **输入**：主文件是扩展名为 `.in` 的 TOML。设置网格、时间、组分与约束、平衡、通量和源模型。文件路径通常相对进程工作目录。不要把模型名、后端模板和径向数据文件当成同一类参数。
2. **计算**：初始化剖面与定值导入；每次试探在同一几何上求源和输运，组装离散残差，接受或回退。默认外部文件只选择一个径向切片；`frame` 是初始化选择，不是随模拟时钟推进的索引。
3. **输出**：`.out` 是终端日志；默认快照 `*.log.npy`（Python pickle，只从可信来源读）；`*.nc` 为可选 NetCDF4（默认关）；`*.bp/` 为 ADIOS2（默认开，可显式关闭写出）。接受步检查点频率由 `[log].n_write` 决定；各输出形态由同一输出 stem 派生而不会共用文件名。

| 数据 | 所在网格与物理单位 | 输入／输出中的名字 |
| --- | --- | --- |
| 组分密度、温度、压强 | 节点；10²⁰ m⁻³、keV、10²⁰ keV m⁻³ | `density`、`temperature`；输出 `n_*`、`T_*`、`p_*` |
| 辅助粒子源、体热源 | 壳层节点；10²⁰ m⁻³ s⁻¹、MW/m³ | `aux_particle_source`、`aux_power_source`；`Sn_*`、`Sp_*` 输出是归一化本地值，乘对应 `Sn_ref_SI20`、`P_ref_MWm3` 转为物理单位 |
| 穿面的粒子率、热功率 | 界面；10²⁰ s⁻¹、MW | `particle_flux`、`heat_flux`；`Gam_SI20_*`、`Q_MW_*` 已有物理单位。不能用 `pflux_*`、`qflux_*`（gyro-Bohm）代替 |
| 时间及有效性 | 时间帧；`t` 以 `t_ref` 归一化 | `t_step_idx`、`t_iter_idx`、`t_diagnostics_valid`；查已接受态时选 `t_iter_idx=0` 并核对有效标志 |

**导入格式。** ASCII `columns` 是空白分列文本（`#` 注释；`divide_by` 按列换单位）；节点初态插值、辅助源区间平均、通量面线性插值是不同操作。T3D `.nc/.npy/.bp` 用 `frame` 取一帧；旧源单位先还原再归一化，通量只读有量纲诊断。若从 T3D 快照恢复 Miller 几何，需保存同一径向网格的 Miller 几何对象的 `.npy`。TRANSP u-file 与 JETTO NetCDF 另可选时间切片用于剖面/源导入，不走通用通量表分支。运行期时变文件读取尚未实现。

**复现与定位。** 从对应源码根目录按 `t3d case.in -l case.out` 或已安装环境中的 `python -m t3d case.in` 启动；主输入不能脱离其相对文件路径。详细[输入参数表](PARAMETERS.md)包括各组默认值、允许值、单位、后端环境和兼容性；用于实验性改模前先核对目标后端的验证边界。

### 相对上游的改动

- **重启单位与网格：** 旧归一化量先还原到物理单位再用新参考量归一化，兼容旧 t_ref；拒绝坐标不匹配、缺测或掩蔽数据；变网格用壳层重叠映射，不声称跨几何体积守恒。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Import.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_source_physics_io.py)
- **文件通量导入：** 新增文件通量模型 Import：复用 [import] 与组分级 import/keys，读取每张磁面的有符号粒子率 (10^20/s) 与热功率 (MW)；在共享通量面线性插值，不外推，不折作体源。支持 ASCII 与已有 T3D 帧，及与 ReLU 相加；固定通量不影响 Newton Jacobian。 [配置](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/docs/FluxModels.rst) · [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/flux_models/Import.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_imported_flux.py)
- **输入解析：** 保留实际 TOML／文件错误，不用错误的字符串回退覆盖原因；ASCII 多列源项求和，兼容 JETTO 时间键。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Import.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_import.py)
- **检查点与输出路径：** 每 N 个接受步写检查点，NPY 和 NetCDF 写完才替换上一份，BP 单独路径；各格式不会共用同一个文件名。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/TrinityIO.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_driver_transactions.py)
- **诊断对齐与单位：** 修正逐物种和离子汇总、粒子量和积分单位，分开输出 α 出生与沉积功率；记录几何文件、时间推进阶数、残差与诊断有效性。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/TrinityIO.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_source_physics_io.py)
- **末帧与后处理：** 内部闭合在最终剖面重算；外部闭合如未同步则明确标 NaN，不为日志隐式启动求解。功率图按所选帧取通量，用已接受状态和已知阶数算时间导数，不猜旧文件的推进阶数。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/tools/source_sink_plot.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_driver_transactions.py)
- **TOML 写回与运行入口：** 字符串、嵌套表和数组可写回并重新读入。源码快照可用 python -m t3d 启动；输入扩展名错误返回非零且不创建输出日志。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/TrinityIO.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_iterative_io.py)
- **TRANSP / JETTO 导入：** TRANSP 用质量和电荷辨认离子，不用电子密度补缺；处理单字符串与多源通道，校验径向网格和单位。JETTO 缺测值不再被当作物理数据。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/Import.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_iterative_io.py)
- **模型标签与失败诊断：** 单表和数组语法用同一诊断标签和计数，检查复合变量名碰撞；计算失败后不保留上一帧的有效诊断标记。 [源码](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/t3d/FluxModels.py) · [测试](https://github.com/Yihui-L/T3D-websites/blob/main/Source-Code/T3D-lyh/tests/unit/test_iterative_io.py)
