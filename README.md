# 📚 Paper Reading Skills for Claude Code

通用学术论文阅读辅助 Skills，适用于各学科研究生，包含两个互补的阅读模式。

## 📦 包含内容

```
paper-reading-skills/
├── read-paper.md          # 论文阅读助手 Skill（交互式引导阅读）
├── paper-mentor.md        # 博导带读文献 Skill（逐段精读+导师反馈）
├── extract_paper.py       # 论文文本提取脚本（支持 PDF/DOCX/TXT/MD）
├── extract_docx.py        # DOCX 专用提取脚本
└── README.md              # 本文件
```

---

## 🎯 两个 Skill 的区别

| 特性 | read-paper（论文阅读助手） | paper-mentor（博导带读文献） |
|------|--------------------------|---------------------------|
| **角色** | 学术阅读导师 | 资深博导（角色扮演） |
| **阅读方式** | 按目的分模块引导 | 逐段精读，不跳段 |
| **适合场景** | 快速了解/定向学习 | 深度理解/批判训练 |
| **时间投入** | 中等（30-60 min） | 较长（1-2 h） |
| **笔记输出** | 125原则笔记 | 逐段精读记录+导师总评 |
| **适用对象** | 本科/硕士/博士/全体教师 | 本科/硕士/博士/全体教师（需要批判训练） |

**建议搭配使用**：先用 `read-paper` 快速筛选和了解论文，对重点论文再用 `paper-mentor` 深度精读。

---

## 🚀 安装步骤

### 1. 环境依赖

需要 Python 3.7+，安装依赖库：

```bash
pip install PyPDF2 python-docx
```

### 2. 安装 Skills

将两个 skill 文件复制到 Claude Code 的 skills 目录：

**Windows (PowerShell)**:
```powershell
copy read-paper.md $env:USERPROFILE\.claude\skills\
copy paper-mentor.md $env:USERPROFILE\.claude\skills\
```

**macOS / Linux**:
```bash
cp read-paper.md ~/.claude/skills/
cp paper-mentor.md ~/.claude/skills/
```

### 3. 放置提取脚本

将 `extract_paper.py` 和 `extract_docx.py` 放在你日常工作的目录下（或添加到系统 PATH），确保在 Claude Code 的工作目录中可以直接调用：

```bash
python extract_paper.py "论文文件路径"
```

### 4. 自定义笔记保存路径（可选）

Skill 中笔记默认保存到 `./notes/` 目录。如需修改，编辑 skill 文件中"第六步：保存笔记"部分的路径。

---

## 📖 使用方法

### read-paper（论文阅读助手）

```
/read-paper <论文文件路径>
```

1. 自动提取论文文本，生成术语预解释表
2. 展示论文速览（中英文对照）
3. 让你选择阅读目的（背景/前沿/方法/写作）
4. 按目的分模块引导阅读，逐层深入提问
5. 生成 125 原则笔记并保存

### paper-mentor（博导带读文献）

```
/paper-mentor <论文文件路径>
```

1. 以博导身份开场，评估你的背景知识
2. 术语预检，逐段精读论文
3. 每段提问 2-3 个拓展问题（四层递进）
4. 逐句反馈你的回答
5. 深度复盘 + 创新点评级 + 导师总评

---

## 🔧 支持的论文格式

- **PDF** — 文字型 PDF（扫描版不支持，需先 OCR）
- **DOCX** — Word 文档
- **TXT** — 纯文本
- **MD** — Markdown 文件

---

## 📝 自定义领域

已去除学科限制，改为**通用学术论文阅读**。Skill 会根据论文实际内容自动适配学科背景。

如需进一步定制（如预设提问方向），可编辑 skill 文件中以下部分：
- 开头的角色描述（可补充你的研究方向）
- 模块 G 中的学科专项问题框架
- `read-paper.md` 中"领域联结"规则的具体内容

---

## 📋 依赖说明

| 文件 | 作用 |
|------|------|
| `extract_paper.py` | 主提取脚本，支持 PDF/DOCX/TXT/MD 四格式 |
| `extract_docx.py` | DOCX 纯文本提取（简易版） |
| `PyPDF2` | Python 库，PDF 文本提取 |
| `python-docx` | Python 库，DOCX 文本提取 |

---

## 📄 许可

自由使用、修改和分享。希望能帮助更多研究生高效阅读文献！
