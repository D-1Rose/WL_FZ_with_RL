from graphviz import Digraph

def draw_basic_fuzzy_logic():
    # 使用 Top-Down (TB) 布局，这种布局画"分支结构"最干净，绝不重叠
    dot = Digraph(comment='Basic Fuzzy Logic', format='png')
    
    # 全局设置：从上到下，正交线
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')
    dot.attr('node', fontname='Helvetica', fontsize='12', shape='rect', style='filled')
    
    # 颜色定义
    COLOR_RL = "#E3F2FD"    # 浅蓝 (参数)
    COLOR_DATA = "#E8F5E9"  # 浅绿 (输入)
    COLOR_LOGIC = "#FFF3E0" # 浅橙 (判决)
    COLOR_EXEC = "#F5F5F5"  # 浅灰 (执行)
    COLOR_OUT = "#FFEBEE"   # 浅红 (输出)

    # ======================================================
    # 1. 顶部：输入源
    # ======================================================
    
    # 我们把 RL 参数放在左上角，作为"配置者"
    dot.node('RL_Input', 'RL 调参指令\n(γ, a, eth)', fillcolor=COLOR_RL, shape='note')
    
    # 把误差输入放在正上方
    dot.node('Err_Input', '传感器原始误差\n(e)', fillcolor=COLOR_DATA, shape='ellipse')

    # ======================================================
    # 2. 中间：逻辑处理核心 (用子图包起来)
    # ======================================================
    with dot.subgraph(name='cluster_Core') as c:
        c.attr(label='S22: 动态误差重塑逻辑', style='dashed', color='blue', fontcolor='blue')
        
        # 判决节点
        c.node('Switch', '区域判决\nCompare |e| with C1, C2', fillcolor=COLOR_LOGIC, shape='diamond')
        
        # 并列的三个执行动作 (使用 rank='same' 强制在同一行)
        with c.subgraph() as actions:
            actions.attr(rank='same')
            actions.node('Zone_C', '中心区 (小误差)\n线性维持\nφ = K0 * e', fillcolor=COLOR_EXEC)
            actions.node('Zone_T', '过渡区 (中误差)\n增益增强\nφ = K1 * e + N1', fillcolor=COLOR_EXEC)
            actions.node('Zone_S', '饱和区 (大误差)\n幅值限制\nSaturation', fillcolor=COLOR_EXEC)
    
    # ======================================================
    # 3. 底部：输出汇聚
    # ======================================================
    dot.node('Output', '虚拟误差\n(φ)', fillcolor=COLOR_OUT, shape='doubleoctagon')

    # ======================================================
    # 4. 连线 (逻辑非常简单)
    # ======================================================
    
    # A. 控制流 (虚线): RL 参数 -> 注入到整个逻辑块
    # lhead='cluster_Core' 需要 graphviz 引擎支持，如果不支持会自动连到最近的节点，也没关系
    dot.edge('RL_Input', 'Switch', label=' 实时修改边界与斜率', style='dashed', color='blue', penwidth='1.5')
    
    # B. 数据流 (实线): 也就是误差怎么走的
    dot.edge('Err_Input', 'Switch', penwidth='2')
    
    # 分支
    dot.edge('Switch', 'Zone_C', label=' Small')
    dot.edge('Switch', 'Zone_T', label=' Mid')
    dot.edge('Switch', 'Zone_S', label=' Large')
    
    # 汇聚
    dot.edge('Zone_C', 'Output')
    dot.edge('Zone_T', 'Output')
    dot.edge('Zone_S', 'Output')

    # 渲染
    dot.render('Basic_Fuzzy_Map', view=False)
    print("生成完毕：Basic_Fuzzy_Map.png")

if __name__ == '__main__':
    draw_basic_fuzzy_logic()