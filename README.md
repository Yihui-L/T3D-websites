# T3D Websites

T3D-lyh 源码手册与 W7-X ReLU 回归算例的公开网站，无需登录。

| 目录 | 内容 | 网站 |
| --- | --- | --- |
| [Source-Code](Source-Code/README.md) | 守恒方程、物理模型、I/O、[完整参数表](Source-Code/PARAMETERS.md)、逐项改动和 [T3D-lyh 源码快照](Source-Code/T3D-lyh/) | [源码手册](https://yihui-l.github.io/T3D-websites/Source-Code/) |
| [W7X-ReLU-lyh](W7X-ReLU-lyh/README.md) | 同一官方输入下的时间演化、两版终态和功率平衡对比、图与数据 | [算例网站](https://yihui-l.github.io/T3D-websites/W7X-ReLU-lyh/) |

```text
T3D-websites/
├── index.html
├── Source-Code/
│   ├── T3D-lyh/
│   └── 源码手册、参数表、版本说明与网页资源
├── W7X-ReLU-lyh/
│   ├── data/
│   └── figures/
├── README.md
└── .nojekyll
```

源码快照的版本和许可见[源码版本说明](Source-Code/SOURCE_VERSION.md)。算例使用固定 VMEC 平衡；发布数据包含官方及 T3D-lyh 的接受态和对比口径，不是实验预测验证。

GitHub Pages 从 `gh-pages` 分支根目录发布。该分支同步根入口、两份文档和算例资源，不复制源码快照；网页源码链接指向 `main` 分支。
