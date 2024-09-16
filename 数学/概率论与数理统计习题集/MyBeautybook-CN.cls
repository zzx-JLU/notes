% This Class is based on the default book class and options will be passed
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MyBeautybook-CN}[2023-06-22 MyBeautybook Class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessOptions\relax
\LoadClass[twoside]{ctexbook}
\overfullrule=1pt

\usepackage[dvipsnames, svgnames, x11names, table]{xcolor} % 颜色扩展
\RequirePackage{bropd} % 数学公式中嵌套括号与微分符号简便输入
\usepackage{tikz-imagelabels} % tikz图像标注
\usepackage{epigraph-keys} % 名言引用
\usepackage{fitbox} % 图像自适应
\usepackage{xeCJKfntef} % 文字两端对齐
\usepackage{pgfornament-han} % 中国风纹样
\usepackage{cncolours} % 中国风颜色
\usepackage{amssymb, amsfonts}
\usepackage{makeidx}
\usepackage{etoolbox} % 判断函数
\usepackage{paracol}
\usepackage{tikz-cd}
\usepackage{imakeidx}
\usepackage{anyfontsize}
\usepackage{calc}
\usepackage{xkeyval, ifthen}
\usepackage{varwidth}
\usepackage[
    colorlinks = true, % 链接文字带颜色
    linkcolor = black, % 链接文字为黑色
    anchorcolor = red,
    urlcolor = blue, % 超链接颜色为蓝色
    citecolor = blue, % 引用颜色为黑色
    bookmarksnumbered = true, % 书签带章节编号
]{hyperref}
\usepackage{dashrule}
\usepackage{caption}
\usepackage{indentfirst}
\usepackage{physics}
\usepackage{amsmath}
\usepackage{zhlipsum}
\usepackage[a4paper, left = 2cm, right = 2cm, top = 2.5cm, bottom = 1.75cm]{geometry}
\usepackage{xpatch} % 修正章节编号
\usepackage{fontawesome5}
\usepackage{mathrsfs}
\usepackage{tabularx}
\usepackage{lastpage}
\usepackage{ninecolors}
\usepackage{colortbl} % 彩色表格
\usepackage{pgfornament}
\usepackage{zhnumber}
\usepackage{adjustbox}
\usepackage{enumitem}
\usepackage{multicol}
\usepackage{amsthm}
\RequirePackage{thmtools, thm-restate} % 定理环境 % 定理的重写
\usepackage{bclogo}
\usepackage{ulem}
\usepackage{graphicx} % 修正minipage顶部对齐问题
\usepackage[explicit]{titlesec}
\usepackage{extarrows}
\usepackage{appendix} % 附录
\usepackage{rotating}
\usepackage{bm, color, framed, bookmark, float, wrapfig}

\usepackage[labelformat=simple]{subcaption}
\renewcommand\thesubfigure{(\alph{subfigure})}

\usepackage{tikz}
\usetikzlibrary{calc, shadows, hobby, intersections, decorations.markings, decorations.pathreplacing, spy, arrows, shapes, fadings, trees, mindmap, patterns, shapes.arrows, shapes.symbols, tikzmark, shapes.geometric, graphs, quotes, angles, decorations.pathmorphing, through, shadings, backgrounds, positioning, fit, arrows.meta, shapes.misc, decorations.shapes, calligraphy}
\tikzset{>=Stealth}

\usepackage[most]{tcolorbox} % 好看的文本框
\tcbuselibrary{breakable, skins, theorems} % TcolorBox Library

\RequirePackage{tabularray}
\UseTblrLibrary{amsmath}
\UseTblrLibrary{booktabs} % 定义 \toprule、\midrule、\bottomrule 和 \cmidrule 命令，这些命令可以直接用于 tblr 环境中
\UseTblrLibrary{diagbox} % 在 tblr 环境中使用 \diagbox 和 \diagboxthree 命令排版斜线表头
\UseTblrLibrary{varwidth} % 为表格添加 measure 内部参数。当设置了 measure=vbox 后，则会使用 \vbox 测量单元格宽度。可以使用 stretch=-1 移除列表环境上下的间距, 注意，stretch=-1 选项也会移除单元格中的支架

\RequirePackage{pgfplots} % 画图。页面样式设计核心包，提供 \pgfonlayer 命令以及下列图层指令
\pgfplotsset{compat = 1.18}
\pgfdeclarelayer{background} % 背景（底层）
\pgfdeclarelayer{foreground} % 上层
\pgfdeclarelayer{top} % 顶部
\pgfdeclarelayer{bottom} % 底部
\pgfsetlayers{bottom, background, main, foreground, top}

\setmainfont{XITS}
\setCJKfamilyfont{zhkai}{KaiTi}[BoldFont=SimHei] % 楷体字的粗体设置为黑体

\setcounter{tocdepth}{2} % 目录深度
\setcounter{secnumdepth}{3} % 编号深度

\setlist{nosep, font=\upshape} % 取消所有列表默认距离

% 公式编号字体保持不变
\makeatletter
\renewcommand{\maketag@@@}[1]{\hbox{\m@th\normalsize\normalfont#1}}%
\makeatother

% 公式编号以 section 为排序单位
\numberwithin{equation}{section}

% \newcommand{\tikzoverlay}[1]{\begin{tikzpicture}[remember picture, overlay] #1 \end{tikzpicture}}

% ------------浮动环境设置
%% 默认情况下, \LaTeX{} 要求每页的文字至少占据 20%, 否则该页就只单独放置一个浮动环境, 而这通常不是我们想要的, 我们将这个要求降低到 5%.
\renewcommand*{\textfraction}{0.05}

%% 有时如果多个浮动环境连续放在一起, 会将它们分在几个不同页, 即使它们可在同一页放得下.
%% 我们可以通过修改 |\topfraction| 和 |\bottomfraction| 分别设置顶端和底端的浮动环境的最大比例.
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}

%% 有时\LaTeX{}会把一个浮动环境单独放在一页,
%% 我们要求这个环境至少要占据 85% 才能单独放在一页.
%% 注意: |\floatpagefraction| 的数值必须小于 |\topfraction|.
\renewcommand*{\floatpagefraction}{0.85}

% 如果图片没有指定后缀, 依次按下列顺序搜索
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
% 设置图表搜索路径, 可以给图表文件夹取如下名字
\graphicspath{{figures/}{figure/}{pictures/}{picture/}{pic/}{pics/}{image/}{images/}{Figures/}}

% tcolorbox 中插入图片
\newcommand\Figure[4]{
    \begin{center}
        \captionsetup{type=figure} % 消除 Package caption Warning: The option 'hypcap=true' will be ignored for this (caption) particular \caption on input line XX. 警告信息
        \includegraphics[#1]{#2}
        \captionof{figure}{#3}
        \label{#4}
    \end{center}
}

% tcolorbox 中插入 tikz 绘图
\newcommand\Tikz[3]{
    \begin{center}
        \captionsetup{type=figure} % 消除 Package caption Warning: The option 'hypcap=true' will be ignored for this(caption) particular \caption on input line XX. 警告信息
        #3 % 最后一个参数为 Tikz 画图代码窗口
        \captionof{figure}{#1}
        \label{#2} % 第一与第二个参数是 Tikz 画图的名称与标签
    \end{center}
}

% 页眉页脚
\usepackage{fancyhdr}
\pagestyle{fancyplain}
\fancyhf{} % 清空页眉页脚
\renewcommand{\headrulewidth}{0.4pt} % 页眉线
\fancyhead[LE,RO]{\thepage} % 偶数页左页眉，奇数页右页眉，页码
\fancyhead[CO]{\rightmark} % 奇数页中页眉
\fancyhead[CE]{\leftmark} % 偶数页中页眉

% plain 页面类型
\fancypagestyle{plain}{
    \fancyhf{} % 清空页眉页脚
    \renewcommand{\headrulewidth}{0pt} % 去页眉线
    \fancyfoot[LE,RO]{\thepage} % 奇数页右页脚，偶数页左页脚，页码
}

\colorlet{outermarginfgcolor}{DarkCyan} % foregroundcolor 较深
\colorlet{outermarginbgcolor}{DarkCyan!30} % backgroundcolor 较浅

% 章节标题设置
\titleformat{\chapter}{\huge\bfseries\filcenter}{第\zhnumber{\arabic{chapter}}章 \quad #1}{0em}{}
% 目录和附录标题设置
\titleformat{name=\chapter, numberless}{\bfseries\huge\filcenter}{#1}{0em}{}

% Part 样式
\makeatletter
\newcommand\partabstract[1]{\def\@partabstract{#1}}
\newcommand\partimage[1]{\def\@partimage{#1}}
\titleformat{\part}{\normalfont\huge\filcenter}{}{20pt}{
    \begin{tikzpicture}[remember picture, overlay]
        \def\barwidth{2cm}
        \fill[outermarginbgcolor, opacity=0.6] (current page.north west) rectangle (current page.south east);
        \ifodd\value{page}
            \fill[outermarginfgcolor] (current page.north east) rectangle ++ (-\barwidth, -\paperheight);
            \node[] (hbar) at ($(current page.north)!0.33!(current page.south)$) {
                \begin{tikzpicture}
                    \fill[white] (0,0) rectangle ++ (\paperwidth, -1cm);
                    \fill[outermarginbgcolor] (0, -.25cm) rectangle ++ (\paperwidth, -.5cm);
                    \fill[outermarginfgcolor] (.67\paperwidth, -.25cm) rectangle ++ (4.5cm, -.5cm);
                    \fill[outermarginfgcolor!20!white] (.67\paperwidth-1mm, -.25cm) rectangle ++ (1mm, -.5cm);
                    \fill[outermarginfgcolor!20!white] (.67\paperwidth+4.4cm, -.25cm) rectangle ++ (1mm, -.5cm);
                \end{tikzpicture}
            };
            \node[above, font=\bfseries\fontsize{30}{30}\selectfont, shift={(.33\linewidth, .5\barwidth)}] (partname) at (hbar) {第\zhnumber{\arabic{part}}部分};
            % 标题名称
            \node[below, left, font=\bfseries\huge, shift={(.146\linewidth, -1.5*\barwidth)}] (partcontents) at (partname) {\begin{varwidth}{.8\linewidth}\raggedright\baselineskip=2.8ex  #1 \end{varwidth}};
            \node[above right, font=\fontsize{15}{15}\selectfont, circle, fill=outermarginfgcolor, text=white, inner sep=1mm] (pagenumber) at ([shift={(5mm, 5mm)}] current page.south west) {\thepage};
        \else
            \fill[outermarginfgcolor] (current page.north west) rectangle ++ (\barwidth, -\paperheight);
            \node[] (hbar) at ($(current page.north)!0.33!(current page.south)$) {
                \begin{tikzpicture}
                    \fill[white] (0,0) rectangle ++ (\paperwidth, -1cm);
                    \fill[outermarginbgcolor] (0,-.25cm) rectangle ++ (\paperwidth, -.5cm);
                    \fill[outermarginfgcolor] (.15\paperwidth,-.25cm) rectangle ++ (4.5cm, -.5cm);
                    \fill[outermarginfgcolor!20!white] (.15\paperwidth-1mm, -.25cm) rectangle ++ (1mm, -.5cm);
                    \fill[outermarginfgcolor!20!white] (.15\paperwidth+4.5cm, -.25cm) rectangle ++ (1mm, -.5cm);
                \end{tikzpicture}
            };
            \node[above, font=\bfseries\huge, shift={(-.31\linewidth, .5\barwidth)}] (partname) at (hbar) {第\,\zhnumber{\arabic{part}}\,部\,分};
            % 标题名称
            \node[below right, font=\bfseries\huge, shift={(-.125\linewidth, -1.2*\barwidth)}] (partcontents) at (partname) {\begin{varwidth}{.8\linewidth}\raggedright\baselineskip=2.8ex  #1 \end{varwidth}};
            \node[above left, font=\fontsize{15}{15}\selectfont, circle, fill=outermarginfgcolor, text=white, inner sep=1mm] (pagenumber) at ([shift={(-5mm, 5mm)}] current page.south east) {\thepage};
        \fi
        % 简介文字调整
        \node[below, font=\kaishu\fontsize{13}{13}\selectfont, shift={(0\linewidth, -1.65*\barwidth)}] at (hbar) {\ifdefvoid{\@partabstract}{}{\begin{varwidth}{.85\linewidth}\baselineskip=3ex \@partabstract\end{varwidth}}};
    \end{tikzpicture}
}
\makeatother

\assignpagestyle{\part}{empty}
\assignpagestyle{\chapter}{empty}

%% Begin The Most Beautiful Box (STYLE : ys style) % 中国风盒子
\makeatletter
\definecolor{framegolden}{HTML}{455a64} % 控制外框颜色
\definecolor{framegray}{HTML}{e0e0e0} % 控制填充颜色
\newdimen\ys@frame@linewidth
\newdimen\ys@frame@radius
\newdimen\ys@frame@sep
\def\ys@frame@fill@opacity{0.2}

\pgfdeclarehorizontalshading{ys@fade@circle}{500bp}{
    color(0bp)=(pgftransparent!100);
    color(25bp)=(pgftransparent!100);
    color(45bp)=(pgftransparent!0);
    color(55bp)=(pgftransparent!0);
    color(75bp)=(pgftransparent!100);
    color(100bp)=(pgftransparent!100)
}
\pgfdeclarefading{ys circle}{\pgfuseshading{ys@fade@circle}}

\def\ys@frame@star@list@t{
    -0.7071\ys@frame@radius + 0.2133\ys@frame@sep/
    - 0.7071\ys@frame@radius + 0.2133\ys@frame@sep/
    frame.north east,
    0.7071\ys@frame@radius - 0.2133\ys@frame@sep/
    -0.7071\ys@frame@radius + 0.2133\ys@frame@sep/
    frame.north west
}
\def\ys@frame@star@list@b{
    0.7071\ys@frame@radius - 0.2133\ys@frame@sep/
    0.7071\ys@frame@radius - 0.2133\ys@frame@sep/
    frame.south west,
    -0.7071\ys@frame@radius + 0.2133\ys@frame@sep/
    0.7071\ys@frame@radius - 0.2133\ys@frame@sep/
    frame.south east
}
\edef\ys@frame@star@list{\ys@frame@star@list@t, \ys@frame@star@list@b}

% region outer
\def\ys@frame@remove@middleline#1#2{}
\def\ys@frame@draw@middleline#1#2{
    \clip ([xshift = -\ys@frame@linewidth, #1] frame.north west) -- ([xshift = \ys@frame@linewidth, #1] frame.north east) -- ([xshift = \ys@frame@linewidth, #2] frame.south east) -- ([xshift = -\ys@frame@linewidth, #2] frame.south west) -- cycle;
}
\def\ys@frame@unbrokenouter{
    \path[draw = framegolden, fill = framegray, fill opacity = \ys@frame@fill@opacity, line width = \ys@frame@linewidth] ([xshift = \ys@frame@radius] frame.north west) -- ([xshift = -\ys@frame@radius] frame.north east) arc (-180:-90:\ys@frame@radius) -- ([yshift = -\ys@frame@radius] frame.north east) -- ([yshift = \ys@frame@radius] frame.south east) arc (90:180:\ys@frame@radius) -- ([xshift = -\ys@frame@radius] frame.south east) -- ([xshift = \ys@frame@radius] frame.south west) arc (0:90:\ys@frame@radius) -- ([yshift = \ys@frame@radius] frame.south west) -- ([yshift = -\ys@frame@radius] frame.north west) arc(-90:0:\ys@frame@radius) -- cycle;
}
\def\ys@frame@firstouter{
    \ys@frame@remove@middleline{yshift=.5\ys@frame@linewidth}{yshift=.7\ys@frame@linewidth}
    \path[draw=framegolden,fill=framegray,fill opacity=\ys@frame@fill@opacity,line width=\ys@frame@linewidth]
    ([xshift=\ys@frame@radius]frame.north west)--([xshift=-\ys@frame@radius]frame.north east)
    arc(-180:-90:\ys@frame@radius)--
    ([yshift=-\ys@frame@radius]frame.north east)--
    (frame.south east)--
    (frame.south west)--
    ([yshift=-\ys@frame@radius]frame.north west)
    arc(-90:0:\ys@frame@radius)--cycle;}
\def\ys@frame@middleouter{
    \ys@frame@remove@middleline{yshift=-.7\ys@frame@linewidth}{yshift=.7\ys@frame@linewidth}
    \path[draw=framegolden,fill=framegray,fill opacity=\ys@frame@fill@opacity,line width=\ys@frame@linewidth]
    (frame.north west)--(frame.north east)--(frame.south east)--
    (frame.south west)--cycle;}
\def\ys@frame@lastouter{
    \ys@frame@remove@middleline{yshift=-.7\ys@frame@linewidth}{yshift=-.5\ys@frame@linewidth}
    \path[draw=framegolden,fill=framegray,fill opacity=\ys@frame@fill@opacity,line width=\ys@frame@linewidth]
    (frame.north west)--(frame.north east)--
    ([yshift=\ys@frame@radius]frame.south east)
    arc(90:180:\ys@frame@radius)--
    ([xshift=-\ys@frame@radius]frame.south east)--([xshift=\ys@frame@radius]frame.south west)
    arc(0:90:\ys@frame@radius)--
    ([yshift=\ys@frame@radius]frame.south west)--
    (frame.north west)--cycle;}
% endregion

% region inner
\def\ys@frame@unbrokeninner{
    \path[draw=framegolden,line width=.5\ys@frame@linewidth,line cap=round]
    ([xshift=\ys@frame@radius+2\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west)
    --([xshift=\ys@frame@radius+5\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west) --++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([xshift=\ys@frame@radius+6\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west)--++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth) % 第一节
    --([xshift=-\ys@frame@radius-8.828\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)--++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth)
    ([xshift=-\ys@frame@radius-7.828\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)--++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) % 第二节
    --([xshift=-\ys@frame@radius-2\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)
    
    ([yshift=-\ys@frame@radius-2\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.north east)
    --([yshift=-\ys@frame@radius-5\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.north east) --++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([yshift=-\ys@frame@radius-6\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.north east)--++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth) % 第一节
    --([yshift=\ys@frame@radius+8.828\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.south east)--++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth)
    ([yshift=\ys@frame@radius+7.828\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.south east)--++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) % 第二节
    
    --([yshift=\ys@frame@radius+2\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.south east)
    
    ([xshift=\ys@frame@radius+2\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west)
    --([xshift=\ys@frame@radius+5\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west) --++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth)
    ([xshift=\ys@frame@radius+6\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west)--++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) % 第一节
    --([xshift=-\ys@frame@radius-8.828\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)--++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([xshift=-\ys@frame@radius-7.828\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)--++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth) % 第二节
    --([xshift=-\ys@frame@radius-2\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)
    
    ([yshift=-\ys@frame@radius-2\ys@frame@linewidth,xshift=\ys@frame@sep]frame.north west)
    --([yshift=-\ys@frame@radius-5\ys@frame@linewidth,xshift=\ys@frame@sep]frame.north west) --++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth)
    ([yshift=-\ys@frame@radius-6\ys@frame@linewidth,xshift=\ys@frame@sep]frame.north west)--++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) % 第一节
    --([yshift=\ys@frame@radius+8.828\ys@frame@linewidth,xshift=\ys@frame@sep]frame.south west)--++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([yshift=\ys@frame@radius+7.828\ys@frame@linewidth,xshift=\ys@frame@sep]frame.south west)--++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth) % 第二节
    --([yshift=\ys@frame@radius+2\ys@frame@linewidth,xshift=\ys@frame@sep]frame.south west);}
\def\ys@frame@firstinner{
    \path[draw=framegolden,line width=.5\ys@frame@linewidth,line cap=round]
    ([xshift=\ys@frame@radius+2\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west)
    --([xshift=\ys@frame@radius+5\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west) --++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([xshift=\ys@frame@radius+6\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west)--++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth) %第一节
    --([xshift=-\ys@frame@radius-8.828\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)--++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth)
    ([xshift=-\ys@frame@radius-7.828\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)--++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) %第二节
    --([xshift=-\ys@frame@radius-2\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)
    %%
    ([yshift=-\ys@frame@radius-2\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.north east)
    --([yshift=-\ys@frame@radius-5\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.north east) --++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([yshift=-\ys@frame@radius-6\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.north east)--++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth) %第一节
    --([xshift=-\ys@frame@sep]frame.south east)

    ([yshift=-\ys@frame@radius-2\ys@frame@linewidth,xshift=\ys@frame@sep]frame.north west)
    --([yshift=-\ys@frame@radius-5\ys@frame@linewidth,xshift=\ys@frame@sep]frame.north west) --++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth)
    ([yshift=-\ys@frame@radius-6\ys@frame@linewidth,xshift=\ys@frame@sep]frame.north west)--++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) %第一节
    --([xshift=\ys@frame@sep]frame.south west);}
\def\ys@frame@middleinner{%
    \path[draw=framegolden,line width=.5\ys@frame@linewidth,line cap=round]
    ([xshift=-\ys@frame@sep]frame.north east)--([xshift=-\ys@frame@sep]frame.south east)
    ([xshift=\ys@frame@sep]frame.north west)--([xshift=\ys@frame@sep]frame.south west);}
\def\ys@frame@lastinner{%
    \path[draw=framegolden,line width=.5\ys@frame@linewidth,line cap=round]
    ([xshift=-\ys@frame@sep]frame.north east)
    --([yshift=\ys@frame@radius+8.828\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.south east)--++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth)
    ([yshift=\ys@frame@radius+7.828\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.south east)--++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) %第二节
    %
    --([yshift=\ys@frame@radius+2\ys@frame@linewidth,xshift=-\ys@frame@sep]frame.south east)
    %
    ([xshift=\ys@frame@radius+2\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west)
    --([xshift=\ys@frame@radius+5\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west) --++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth)
    ([xshift=\ys@frame@radius+6\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west)--++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth) %第一节
    --([xshift=-\ys@frame@radius-8.828\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)--++(45:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([xshift=-\ys@frame@radius-7.828\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)--++(-45:2\ys@frame@linewidth)--++(45:2\ys@frame@linewidth) %第二节
    --([xshift=-\ys@frame@radius-2\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)
    %
    ([xshift=\ys@frame@sep]frame.north west)
    --([yshift=\ys@frame@radius+8.828\ys@frame@linewidth,xshift=\ys@frame@sep]frame.south west)--++(-135:2\ys@frame@linewidth)--++(-45:2\ys@frame@linewidth)
    ([yshift=\ys@frame@radius+7.828\ys@frame@linewidth,xshift=\ys@frame@sep]frame.south west)--++(-45:2\ys@frame@linewidth)--++(-135:2\ys@frame@linewidth) %第二节
    --([yshift=\ys@frame@radius+2\ys@frame@linewidth,xshift=\ys@frame@sep]frame.south west);}
%endregion

%region corner
\def\ys@frame@unbrokencorner{
    \path[draw=framegolden,line width=1.5\ys@frame@linewidth,line cap=round]
    ([xshift=\ys@frame@radius-2.7\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west)
    --([xshift=\ys@frame@sep,yshift=-\ys@frame@sep]frame.north west)
    --([xshift=\ys@frame@sep,yshift=-\ys@frame@radius+2.7\ys@frame@linewidth]frame.north west)
    ([xshift=-\ys@frame@radius+2.7\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)
    --([xshift=-\ys@frame@sep,yshift=-\ys@frame@sep]frame.north east)
    --([xshift=-\ys@frame@sep,yshift=-\ys@frame@radius+2.7\ys@frame@linewidth]frame.north east)
    ([xshift=-\ys@frame@radius+2.7\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)
    --([xshift=-\ys@frame@sep,yshift=\ys@frame@sep]frame.south east)
    --([xshift=-\ys@frame@sep,yshift=\ys@frame@radius-2.7\ys@frame@linewidth]frame.south east)
    ([xshift=\ys@frame@radius-2.7\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west)
    --([xshift=\ys@frame@sep,yshift=\ys@frame@sep]frame.south west)
    --([xshift=\ys@frame@sep,yshift=\ys@frame@radius-2.7\ys@frame@linewidth]frame.south west);}
\def\ys@frame@firstcorner{
    \path[draw=framegolden,line width=1.5\ys@frame@linewidth,line cap=round]
    ([xshift=\ys@frame@radius-2.7\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north west)
    --([xshift=\ys@frame@sep,yshift=-\ys@frame@sep]frame.north west)
    --([xshift=\ys@frame@sep,yshift=-\ys@frame@radius+2.7\ys@frame@linewidth]frame.north west)
    ([xshift=-\ys@frame@radius+2.7\ys@frame@linewidth,yshift=-\ys@frame@sep]frame.north east)
    --([xshift=-\ys@frame@sep,yshift=-\ys@frame@sep]frame.north east)
    --([xshift=-\ys@frame@sep,yshift=-\ys@frame@radius+2.7\ys@frame@linewidth]frame.north east);}
\def\ys@frame@middlecorner{}
\def\ys@frame@lastcorner{
    \path[draw=framegolden,line width=1.5\ys@frame@linewidth,line cap=round]
    ([xshift=-\ys@frame@radius+2.7\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south east)
    --([xshift=-\ys@frame@sep,yshift=\ys@frame@sep]frame.south east)
    --([xshift=-\ys@frame@sep,yshift=\ys@frame@radius-2.7\ys@frame@linewidth]frame.south east)
    ([xshift=\ys@frame@radius-2.7\ys@frame@linewidth,yshift=\ys@frame@sep]frame.south west)
    --([xshift=\ys@frame@sep,yshift=\ys@frame@sep]frame.south west)
    --([xshift=\ys@frame@sep,yshift=\ys@frame@radius-2.7\ys@frame@linewidth]frame.south west);}
%endregion

\def\ys@frame@star#1#2#3{
    \begin{scope}[shift={([xshift=#1,yshift=#2]#3)},scale=\ys@frame@ratio*0.1]
        \path[draw=framegolden,fill=white,
            line width=.7\ys@frame@linewidth,line join=round]
        (1,1)..controls(0.3,0.5) and (-0.3,0.5)..
        (-1,1)..controls(-0.5,0.3) and (-0.5,-0.3)..
        (-1,-1)..controls(-0.3,-0.5) and (0.3,-0.5)..
        (1,-1)..controls(0.5,-0.3) and (0.5,0.3)..(1,1)--cycle;
        \path[scale=0.45,fill=framegolden,line join=round]
        (1,1)..controls(0.3,0.7) and (-0.3,0.7)..
        (-1,1)..controls(-0.7,0.3) and (-0.7,-0.3)..
        (-1,-1)..controls(-0.3,-0.7) and (0.3,-0.7)..
        (1,-1)..controls(0.7,-0.3) and (0.7,0.3)..(1,1)--cycle;
    \end{scope}}
\def\ys@frame@segmentation{
    \draw[color=framegolden,path fading=ys circle]
    ([xshift=5\ys@frame@radius]segmentation.west) to[ornament=88]
    ([xshift=-5\ys@frame@radius]segmentation.east);}

\tcbset{
    % 控制是否在断点处的线不画出来
    ys/frame/not at break/.code = \let\ys@frame@remove@middleline\ys@frame@draw@middleline,
    ys/frame/fill opacity/.store in=\ys@frame@fill@opacity,
    ys/frame/draw at break/.code=\let\ys@frame@remove@middleline\@gobbletwo,
    % 控制线宽
    ys/frame/line width/.code={\pgfmathsetlength\ys@frame@linewidth{#1}},
    ys/frame/line width=1.4pt,
    % 控制圆弧大小，应为数字，而非长度
    ys/frame/radius/.code={\pgfmathparse{#1}\let\ys@frame@ratio\pgfmathresult
            \pgfmathsetlength\ys@frame@radius{(#1)*1em}%
            \pgfmathsetlength\ys@frame@sep{(#1)*0.5ex}},
    ys/frame/radius=2,
    ys/frame/unbroken/.style={
            interior code={
                    \ys@frame@unbrokenouter % 外框
                    \foreach \ii/\jj/\kk in \ys@frame@star@list {\ys@frame@star{\ii}{\jj}{\kk}} % 星
                    \ys@frame@unbrokeninner % 内线
                    \ys@frame@unbrokencorner % 四角
                },
            interior titled code={
                    \ys@frame@unbrokenouter % 外框
                    \foreach \ii/\jj/\kk in \ys@frame@star@list {\ys@frame@star{\ii}{\jj}{\kk}} % 星
                    \ys@frame@unbrokeninner % 内线
                    \ys@frame@unbrokencorner % 四角
                }
        },
    ys/frame/first/.style={
            interior code={
                    \ys@frame@firstouter
                    \foreach \ii/\jj/\kk in \ys@frame@star@list@t {\ys@frame@star{\ii}{\jj}{\kk}}
                    \ys@frame@firstinner
                    \ys@frame@firstcorner},
            interior titled code={
                    \ys@frame@firstouter
                    \foreach \ii/\jj/\kk in \ys@frame@star@list@t {\ys@frame@star{\ii}{\jj}{\kk}}
                    \ys@frame@firstinner
                    \ys@frame@firstcorner},
        },
    ys/frame/middle/.style={
            interior code={
                    \ys@frame@middleouter
                    \ys@frame@middleinner
                    \ys@frame@middlecorner}
        },
    ys/frame/last/.style={
            interior code={
                    \ys@frame@lastouter
                    \foreach \ii/\jj/\kk in \ys@frame@star@list@b {\ys@frame@star{\ii}{\jj}{\kk}}
                    \ys@frame@lastinner
                    \ys@frame@lastcorner}
        },
    ys/frame style/unbroken/.style={
            boxsep=2\ys@frame@sep,
            left={.8\ys@frame@radius},right={.8\ys@frame@radius},
            ys/frame/unbroken,title hidden,
            segmentation code=\ys@frame@segmentation
        },
    ys/frame style/first/.style={ys/frame/first,title hidden,segmentation code=\ys@frame@segmentation},
    ys/frame style/middle/.style={ys/frame/middle,segmentation code=\ys@frame@segmentation},
    ys/frame style/last/.style={ys/frame/last,segmentation code=\ys@frame@segmentation},
    ys style/.style={frame hidden,
            ys/frame style/unbroken,
            extras first={ys/frame style/first},
            extras middle={ys/frame style/middle},
            extras last={ys/frame style/last},
        },
}
\makeatother
%% End The Most Beautiful Box

% 键值设置
\makeatletter
\pgfkeys{/mytcbtheorem/.is family}
\newcommand{\mytcbtheoremset}{\pgfqkeys{/mytcbtheorem}}
\mytcbtheoremset{
    name/.code=\@namedef{\my@curr@thmenvname @name}{#1}, % 无效
    counter/.code=\@namedef{my@\my@curr@thmenvname @counter}{#1}\newcounter{#1}[section], % 计数器遇到subsection就会自动归零
    the counter/.code=\@namedef{the\@nameuse{my@\my@curr@thmenvname @counter}}{#1},
    autoref name/.code=\@namedef{\@nameuse{my@\my@curr@thmenvname @counter}autorefname}{#1},
    style/.code=\tcbset{my@\my@curr@thmenvname @tcbthmstyle/.style={#1}},
    thmcolor/.code=\@namedef{\my@curr@thmenvname @thmcolor}{#1}, % 无效
    lemcolor/.code=\@namedef{\my@curr@thmenvname @lemcolor}{#1}, % 无效
}

% 存储键值 (LaTeX3)
\ExplSyntaxOn
\cs_new_protected:Npn \mynewtheorem
{ \keyval_parse:NNn \@my@newtheorem@n \@my@newtheorem@nn }
\cs_new_protected:Npn \mynewtcbtheorem
{ \keyval_parse:NNn \@my@newtcbtheorem@n \@my@newtcbtheorem@nn }
\ExplSyntaxOff
\makeatother

% 第一种定理
\makeatletter
\tcbset{my@basic@thm/.style={enhanced jigsaw, breakable,
            enlarge left by=-3.5mm, width=\textwidth+3.5mm, boxrule=0pt,
            top=2pt, bottom=2pt, left=2.5mm, frame hidden}}
\newcommand{\@my@newtheorem@n}[1]{\ERROR}
\newcommand{\@my@newtheorem@nn}[2]{
    \@ifundefined{#1}{\@my@newtheorem{#1}#2}{\@my@renewtheorem{#1}#2}}
\NewDocumentCommand{\@my@newtheorem}{ m m o +m }
{\IfNoValueTF{#3}{\newtheorem{#1}{#2}}{\newtheorem{#1}{#2}[#3]}
    \tcbset{my@#1@thmstyle/.style={#4}}
    \tcolorboxenvironment{#1}{my@basic@thm, my@#1@thmstyle}}
\NewDocumentCommand{\@my@renewtheorem}{ m m o +m }
{\IfNoValueTF{#3}{\renewtheorem{#1}{#2}}{\renewtheorem{#1}{#2}[#3]}
    \tcbset{my@#1@thmstyle/.style={#4}}}
\makeatother

%% 定义第一种定理
\definecolor{CyaN}{HTML}{dbe1e5}
\colorlet{OrangE}{yellow!20!orange}
\colorlet{BluE}{cyan!70!blue}
\colorlet{ReD}{red!20!orange}
\colorlet{GreeN}{yellow!40!green}

\mynewtheorem{
    defi={\textbf{定义}}[section]{interior style={left color=ReD!8,right color=ReD!5!CyaN!50}, borderline west={1.5mm}{0mm}{ReD}},
    thm={\textbf{定理}}[section]{interior style={left color=CyaN!80!black!20,right color=CyaN!80!black!15!CyaN!50}, borderline west={1.5mm}{0mm}{CyaN!80!black}},
    lem={\textbf{引理}}[section]{interior style={left color=BluE!8,right color=BluE!5!CyaN!50}, borderline west={1.5mm}{0mm}{BluE}},
    prop={\textbf{命题}}[section]{interior style={left color=OrangE!8,right color=OrangE!5!CyaN!50}, borderline west={1.5mm}{0mm}{OrangE}},
    exam={\textbf{题}}[chapter]{interior style={left color=DarkGreen!8,right color=DarkGreen!5!CyaN!50}, borderline west={1.5mm}{0mm}{DarkGreen}},
    cor={\textbf{推论}}[chapter]{interior style={left color=violet!8,right color=violet!5!CyaN!50}, borderline west={1.5mm}{0mm}{violet}},
}
\newtheorem*{remark}{\textbf{注}}

% 第二种定理
\makeatletter
\tcbset{my@basic@tcbthm/.style={detach title, parbox=false}}
\newcommand{\@my@newtcbtheorem@n}[1]{\ERROR}
\newcommand{\@my@newtcbtheorem@nn}[2]{
    \@ifundefined{#1}{\@my@newtcbtheorem{#1}{#2}}{\@my@renewtcbtheorem{#1}{#2}}}
\newcommand{\@my@newtcbtheorem}[2]{
    \edef\my@curr@thmenvname{#1} % 它保存当前定理环境的名称
    \mytcbtheoremset{#2}
    \NewTColorBox{#1}{ O{} O{} o }
    {my@basic@tcbthm, my@#1@tcbthmstyle, title={##2}, IfNoValueF={##3}{label={##3}}, ##1}}
\newcommand{\@my@renewtcbtheorem}[2]{\edef\my@curr@thmenvname{#1}\mytcbtheoremset{#2}}

% 第二种定理，第一类样式
%%  Overlay Settings
\newcommand{\my@theorem@overlay@first}[2]{
    \fill[black!60] ([xshift=1cm,yshift=5pt]frame.north west)-- ([xshift=0.8cm,yshift=0cm]frame.north west)--([xshift=1.2cm,yshift=0cm]frame.north west)--cycle;
    \node[rectangle, text=white, drop shadow={opacity=.3, shadow xshift=0.1cm}, inner sep=1.5mm,fill=#2, anchor=west,rounded corners=3pt,font=\bfseries\normalsize] at ([xshift=0cm,yshift=-3.mm]frame.north west) {\hspace*{1.5em}#1\ \ifdefvoid{\tcbtitletext}{}{(\tcbtitletext)}};
    \fill[color=#2,drop shadow={opacity=0.3,shadow xshift=.3pt}]
    ([xshift=1cm,yshift=5pt]frame.north west) {[rounded corners=3pt]--++(-1cm-5pt,0)}--++(0,-1cm-5pt)--cycle;
    \fill[color=black!60] ([xshift=-5pt,yshift=-1cm]frame.north west)--([xshift=0cm,yshift=-0.8cm]frame.north west)-- ([xshift=0cm,yshift=-1.2cm]frame.north west)--cycle;
}
\newcommand{\my@theorem@overlay@last}[1]{
    \fill[#1,drop shadow={opacity=0.3,shadow xshift=-0.008\linewidth,shadow yshift=0.0016\linewidth}]
    ([xshift=-0.05\linewidth,yshift=-5pt]frame.south east) {[rounded corners=3pt]--++(0.05\linewidth+5pt,0)}--++(0,0.05\linewidth+5pt)--cycle;
    \fill[black!60] ([xshift=-0.05\linewidth,yshift=-5pt]frame.south east) -- ([xshift=-0.04\linewidth,yshift=0pt]frame.south east)-- ([xshift=-0.06\linewidth,yshift=0pt]frame.south east)--cycle;
    \fill[black!60] ([xshift=5pt,yshift=0.05\linewidth]frame.south east)-- ([yshift=0.04\linewidth]frame.south east) -- ([yshift=0.06\linewidth]frame.south east)--cycle;
}
\newcommand{\my@theorem@overlay@unbroken}[2]{\my@theorem@overlay@first{#1}{#2}}

% 第二种定理，第二类样式
\newcommand{\my@lemma@overlay@first}[2]{
    \def\Rad{3.5pt}
    \def\theight{0.57cm}
    \def\rad{7pt}
    \path[fill=#2,drop shadow={opacity=0.3,shadow xshift=.05cm,shadow yshift=-.05cm}]node[append after command={
                ([xshift=-1.5*\rad]thmname.north west) to[out=0,in=180,looseness=1] ([xshift=2.5*\rad]thmname.south west) --([xshift=-\rad]thmname.south east) to[out=0,in=180,looseness=1] ([xshift=3*\rad]thmname.north east) --cycle},
        text=white,font=\bfseries,align=center,inner ysep=1.5mm,minimum height=0.6cm,right] (thmname) at ([xshift=\rad,yshift=-\rad]frame.north west) {\hspace*{10pt}#1\ \ifdefvoid{\tcbtitletext}{}{(\tcbtitletext)}};
    \begin{pgfonlayer}{bottom}
        \fill[#2,opacity=0.5]
        ([xshift=-1.5*\rad]thmname.north west) to[out=-90,in=0,looseness=1.5] +(3pt,-6pt) -| +(10pt,0)--cycle;
        \fill[#2,opacity=0.5]
        ([xshift=3*\rad]thmname.north east) to[out=0,in=-90] +(5pt,-5pt) -| ++(-10pt,0) --cycle;
    \end{pgfonlayer}
}
\newcommand{\my@lemma@overlay@last}[1]{
    \def\Rad{3.5pt}
    \def\rad{7pt}
    \def\theight{0.57cm}
    \def\tlen{1.5cm}
    \path[fill=#1,drop shadow={opacity=0.3,shadow xshift=.3pt}]
    ([xshift=-2*\rad,yshift=-5pt]frame.south east) coordinate (SE)  to[out=0,in=180] +(4*\rad,5pt+\theight) --([shift={(-\tlen+4*\rad,5pt+\theight)}]SE) to[out=180,in=0] +(-4*\rad,-5pt-\theight) --cycle;
    \begin{pgfonlayer}{bottom}
        \fill[color=#1!80!black] ([xshift=-1.35*\tlen+\Rad,yshift=-5pt]frame.south east) to[out=180,in=90,looseness=0.7] ([xshift=-1.35*\tlen-0.5*\Rad]frame.south east) coordinate (NWl)--([xshift=2*\rad]NWl) to[out=-90,in=-180,looseness=0.7] ([xshift=-1.35*\tlen+\Rad,yshift=-5pt]frame.south east)--cycle;
    \end{pgfonlayer}
}
\newcommand{\my@lemma@overlay@unbroken}[2]{\my@lemma@overlay@first{#1}{#2}}
\makeatother

% 定义第二种定理
\definecolor{高粱红}{HTML}{951c48}

\makeatletter
\mynewtcbtheorem{
    % theorem 是环境名
    theorem={
            counter=tcbthm,
            the counter=\thesection.\arabic{tcbthm},
            name=定理, % 它保存到 \theorem@name 里
            thmcolor=高粱红,
            autoref name=\bfseries 定理,
            style={
                    arc=3pt,breakable,enhanced,
                    interior style={top color=高粱红!12, middle color=高粱红!9, bottom color=高粱红!6},
                    boxrule=0pt,top=8mm,
                    fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray}, % 上
                    fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray}, % 下
                    opacityframe=0, opacityback=0.98,
                    fontupper=\kaishu, step={tcbthm},
                    before pre=\smallskip, after app=\smallskip,
                    overlay unbroken=\my@theorem@overlay@unbroken{\theorem@name\ \thetcbthm}{\theorem@thmcolor},
                    overlay first = \my@theorem@overlay@first{\theorem@name\ \thetcbthm}{\theorem@thmcolor}
                }
        },
    proposition={
            counter=tcbprop,
            the counter=\thesection.\arabic{tcbprop},
            autoref name=\bfseries 命题,
            style={
                    arc=3pt,breakable,enhanced,interior style={top color=杏红!12 ,middle color=杏红!9, bottom color=杏红!6},
                    boxrule=0pt,top=8mm,
                    fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray},% 上
                    fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray},%下
                    opacityframe=0, opacityback=0.98,
                    fontupper=\kaishu, step={tcbprop},
                    before pre=\smallskip, after app=\smallskip,
                    overlay unbroken=\my@theorem@overlay@unbroken{命题\ \thetcbprop}{杏红},
                    overlay first = \my@theorem@overlay@first{命题\ \thetcbprop}{杏红}
                }
        },
    definition={
            counter=tcbdefi,
            the counter=\thesection.\arabic{tcbdefi},
            autoref name=\bfseries 定义,
            style={
                    arc=3pt,breakable,enhanced,interior style={top color=紫棠!12 ,middle color=紫棠!9, bottom color=紫棠!6},
                    boxrule=0pt,top=8mm,
                    fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray},% 上
                    fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray},%下
                    opacityframe=0, opacityback=0.98,
                    fontupper=\kaishu, step={tcbdefi},
                    before pre=\smallskip, after app=\smallskip,
                    overlay unbroken=\my@theorem@overlay@unbroken{定义\ \thetcbdefi}{紫棠},
                    overlay first = \my@theorem@overlay@first{定义\ \thetcbdefi}{紫棠}
                }
        },
    lemma={
            counter=tcblem,
            the counter=\thesection.\arabic{tcblem},
            name=引理,
            lemcolor=靛蓝,
            autoref name=\bfseries 引理,
            style={
                    arc=3pt,breakable,enhanced,interior style={top color=靛蓝!12 ,middle color=靛蓝!9, bottom color=靛蓝!6},
                    boxrule=0pt,top=8mm,
                    fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray},% 上
                    fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray},%下
                    opacityframe=0, opacityback=0.98,
                    fontupper=\kaishu,step={tcblem},
                    before pre=\smallskip, after app=\smallskip,
                    overlay unbroken=\my@theorem@overlay@unbroken{\lemma@name\ \thetcblem}{\lemma@lemcolor},
                    overlay first = \my@theorem@overlay@first{\lemma@name\ \thetcblem}{\lemma@lemcolor}
                }
        },
    corollary={
            counter=tcbcor,
            the counter=\thesection.\arabic{tcbcor},
            autoref name=\bfseries 推论,
            style={
                    arc=3pt,breakable,enhanced,interior style={top color=茶色!12 ,middle color=茶色!9, bottom color=茶色!6},
                    boxrule=0pt,top=8mm,
                    fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray},% 上
                    fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray},%下
                    opacityframe=0, opacityback=0.98,
                    fontupper=\kaishu,step={tcbcor},
                    before pre=\smallskip, after app=\smallskip,
                    overlay unbroken=\my@theorem@overlay@unbroken{推论\ \thetcbcor}{茶色},
                    overlay first = \my@theorem@overlay@first{推论\ \thetcbcor}{茶色}
                }
        },
    example = {
            counter=tcbexam,
            the counter=\thesection.\arabic{tcbexam},
            autoref name=\bfseries 例题,
            style={
                    arc=3pt,breakable,enhanced,interior style={top color=黛绿!12 ,middle color=黛绿!9, bottom color=黛绿!6},
                    boxrule=0pt,top=8mm,
                    fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray}, % 上
                    fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray}, % 下
                    opacityframe=0, opacityback=0.98,
                    fontupper=\normalsize,step={tcbexam},
                    before pre=\smallskip, after app=\smallskip,
                    overlay unbroken=\my@lemma@overlay@unbroken{例题\ \thetcbexam}{黛绿},
                    overlay first = \my@lemma@overlay@first{例题\ \thetcbexam}{黛绿}
                }
    },
    conclusion = {
        counter = tcbcon,
        the counter = \arabic{tcbcon},
        thmcolor = 藏蓝,
        autoref name = \bfseries 结论,
        style = {
                arc=3pt,breakable,enhanced,
                interior style={top color=藏蓝!12, middle color=藏蓝!9, bottom color=藏蓝!6},
                boxrule=0pt,top=8mm,
                fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray}, % 上
                fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray}, % 下
                opacityframe=0, opacityback=0.98,
                fontupper=\kaishu, step={tcbcon},
                before pre=\smallskip, after app=\smallskip,
                overlay unbroken=\my@theorem@overlay@unbroken{结论}{藏蓝},
                overlay first = \my@theorem@overlay@first{结论}{藏蓝}
        }
    },
    property = {
        counter = tcbproperty,
        the counter = \thesection.\arabic{tcbproperty},
        thmcolor = 赤金,
        autoref name = \bfseries 性质,
        style = {
                arc=3pt,breakable,enhanced,
                interior style={top color=赤金!12, middle color=赤金!9, bottom color=赤金!6},
                boxrule=0pt,top=8mm,
                fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray}, % 上
                fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray}, % 下
                opacityframe=0, opacityback=0.98,
                fontupper=\kaishu, step={tcbproperty},
                before pre=\smallskip, after app=\smallskip,
                overlay unbroken=\my@theorem@overlay@unbroken{性质 \thetcbproperty}{赤金},
                overlay first = \my@theorem@overlay@first{性质 \thetcbproperty}{赤金}
        }
    },
    note = {
        counter = tcbnote,
        the counter = \arabic{tcbnote},
        thmcolor = 灰色,
        autoref name = \bfseries 备注,
        style = {
                arc=3pt,breakable,enhanced,
                interior style={top color=灰色!12, middle color=灰色!9, bottom color=灰色!6},
                boxrule=0pt,top=8mm,
                fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray}, % 上
                fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray}, % 下
                opacityframe=0, opacityback=0.98,
                fontupper=\normalsize, step={tcbnote},
                before pre=\smallskip, after app=\smallskip,
                overlay unbroken=\my@lemma@overlay@unbroken{备注}{灰色},
                overlay first = \my@lemma@overlay@first{备注}{灰色}
        }
    },
    axiom = {
        counter = tcbax,
        the counter = \thesection.\arabic{tcbax},
        thmcolor = 乌黑,
        autoref name = \bfseries 公理,
        style = {
                arc=3pt,breakable,enhanced,
                interior style={top color=乌黑!12, middle color=乌黑!9, bottom color=乌黑!6},
                boxrule=0pt,top=8mm,
                fuzzy shadow={-0.6mm}{0.6mm}{0mm}{0.3mm}{white!50!gray}, % 上
                fuzzy shadow={0.6mm}{-0.6mm}{0mm}{0.3mm}{fill=white!40!gray}, % 下
                opacityframe=0, opacityback=0.98,
                fontupper=\kaishu, step={tcbax},
                before pre=\smallskip, after app=\smallskip,
                overlay unbroken=\my@theorem@overlay@unbroken{公理 \thetcbax}{乌黑},
                overlay first = \my@theorem@overlay@first{公理 \thetcbax}{乌黑}
        }
    }
}
\makeatother

% 证明环境
\newtcolorbox{Proof}{
    title = 证明,
    fonttitle = \bfseries,
    coltitle = black,
    attach title to upper = {：},
    breakable,
    arc = 0mm,
    outer arc = 0mm,
    parbox = false,
    after upper = \qed,
    colback = white
}
\renewenvironment{proof}{\begin{Proof}}{\end{Proof}}

% 题解环境
\newtcolorbox{solution}{
    title = 解,
    fonttitle = \bfseries,
    coltitle = black,
    attach title to upper = {：},
    breakable,
    arc = 0mm,
    outer arc = 0mm,
    parbox = false,
    colback = white
}

% 分析环境
\newtcolorbox{analysis}{
    title = 分析,
    fonttitle = \bfseries,
    coltitle = black,
    attach title to upper = {：},
    breakable,
    arc = 0mm,
    outer arc = 0mm,
    parbox = false
}

% 习题
\newcounter{questionNB}[section]
\renewcommand{\thequestionNB}{\thesection.\arabic{questionNB}}
\newcommand{\question}{
    \par
    \refstepcounter{questionNB}
    \textbf{\thequestionNB}\hspace{1em}
}

% 参考文献
\usepackage[
    backend = biber,
    style = gb7714-2015,
    sorting = nty,
    isbn = false
]{biblatex}
\indexsetup{level = \chapter*, noclearpage}
\makeindex[title = 索引, columns = 3, columnsep = 15pt, columnseprule]
\makeindex

% 列表环境的格式
\setenumerate[1]{itemsep=0pt, partopsep=0pt, parsep=\parskip, topsep=0pt, left=2em}
\setitemize[1]{itemsep=0pt, partopsep=0pt, parsep=\parskip, topsep=0pt, left=2em}
\setdescription{itemsep=0pt, partopsep=0pt, parsep=\parskip, topsep=0pt}

% tikz 图形样式定义
\tikzset{
    % 空心圆点
    dot/.style = {
        draw,
        fill = white,
        circle,
        inner sep = 0pt,
        minimum size = 4pt,
    }
}

\pgfplotsset{
    2d plot axis/.style = {
        axis lines = center,
        xlabel=$x$, ylabel=$y$,
        every axis x label/.style = {
            at = {(xticklabel* cs:1)},
            anchor = north
        },
        every axis y label/.style = {
            at = {(yticklabel* cs:1)},
            anchor = east
        },
        no markers,
        every axis plot/.append style = {line width=1.5pt},
        tick align = inside,
        extra x ticks = {0},
        extra x tick labels = {$O$},
        extra x tick style = {
            tick label style = {anchor=north east}
        },
        legend style = {
            cells = {anchor=west}
        }
    }
}

\captionsetup[table]{labelsep=quad} % 表格标题的分隔符
\captionsetup[figure]{labelsep=quad} % 图片标题的分隔符