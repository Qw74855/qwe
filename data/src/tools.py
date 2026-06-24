from datetime import datetime

def get_current_week():
    """获取当前是第几周（校历，每年9月1日为第一周起点）"""
    today = datetime.now()
    # 自动判断当年9月1日，适配2026及后续年份
    start_year = today.year if today.month >= 9 else today.year - 1
    term_start = datetime(start_year, 9, 1)
    day_diff = (today - term_start).days
    week_num = day_diff // 7 + 1
    return f"现在是第{week_num}周"

def calculate_gpa(scores_str):
    """
    计算平均绩点
    :param scores_str: 逗号分隔数字字符串，例：'85,90,78'
    """
    # 去除首尾空格并判空
    scores_str = scores_str.strip()
    if not scores_str:
        return "错误：未输入任何成绩"
    try:
        score_list = [int(x.strip()) for x in scores_str.split(',')]
    except ValueError:
        return "错误：成绩必须为数字，使用英文逗号分隔"

    total_gpa = 0.0
    for score in score_list:
        # 分数合法性校验
        if not (0 <= score <= 100):
            return f"错误：存在无效分数 {score}，分数范围0~100"
        if score >= 90:
            total_gpa += 4.0
        elif score >= 80:
            total_gpa += 3.0
        elif score >= 70:
            total_gpa += 2.0
        elif score >= 60:
            total_gpa += 1.0
        else:
            total_gpa += 0.0
    avg_gpa = total_gpa / len(score_list)
    return f"您的平均绩点是：{avg_gpa:.2f}"
