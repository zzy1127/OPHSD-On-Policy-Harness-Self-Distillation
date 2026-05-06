"""Lightweight evaluation for LawBench task 3-3 (accusation prediction).

Metric: macro-averaged F1 over prediction/ground-truth *sets* of accusations.
Exactly mirrors ``test/LawBench/evaluation/evaluation_functions/ljp_accusation.py``
but can run standalone.
"""

# fmt: off
# Extracted from training data (data_train.json) – 202 unique accusation types.
# The original LawBench option list had 189 entries; 13 labels present in the
# actual data were missing and are now included.
OPTION_LIST: list[str] = [
    # ── 伪造 / 变造 / 倒卖 ──
    "伪造、倒卖伪造的有价票证",
    "伪造、变造居民身份证",
    "伪造、变造金融票证",
    "伪造、变造、买卖国家机关公文、证件、印章",
    "伪造、变造、买卖武装部队公文、证件、印章",
    "伪造公司、企业、事业单位、人民团体印章",
    "伪造货币",
    "伪证",
    # ── 出售 / 购买 / 运输 ──
    "出售、购买、运输假币",
    # ── 制作 / 制造 ──
    "制作、复制、出版、贩卖、传播淫秽物品牟利",
    "制造、贩卖、传播淫秽物品",
    # ── 引诱 / 容留 ──
    "引诱、容留、介绍卖淫",
    "引诱、教唆、欺骗他人吸毒",
    # ── 持有 / 使用 ──
    "持有、使用假币",
    "持有伪造的发票",
    # ── 掩饰 / 隐瞒 ──
    "掩饰、隐瞒犯罪所得、犯罪所得收益",
    # ── 生产 / 销售 ──
    "生产、销售有毒、有害食品",
    "生产、销售不符合安全标准的食品",
    "生产、销售伪劣农药、兽药、化肥、种子",
    "生产、销售伪劣产品",
    "生产、销售假药",
    # ── 盗窃 / 侮辱 / 抢夺 ──
    "盗窃、侮辱尸体",
    "盗窃、抢夺枪支、弹药、爆炸物",
    "盗窃、抢夺枪支、弹药、爆炸物、危险物质",
    # ── 窃取 / 收买 ──
    "窃取、收买、非法提供信用卡信息",       # ← previously missing
    # ── 窝藏 / 包庇 / 转移 ──
    "窝藏、包庇",
    "窝藏、转移、收购、销售赃物",            # ← previously missing
    "窝藏、转移、隐瞒毒品、毒赃",
    # ── 组织 ──
    "组织、强迫、引诱、容留、介绍卖淫",
    "组织、领导传销活动",
    "组织、领导、参加黑社会性质组织",
    "组织卖淫",
    # ── 编造 ──
    "编造、故意传播虚假恐怖信息",
    # ── 虚开 ──
    "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票",
    "虚开发票",
    "虚报注册资本",
    # ── 走私 ──
    "走私",                                  # ← previously missing
    "走私、贩卖、运输、制造毒品",
    "走私武器、弹药",
    "走私珍贵动物、珍贵动物制品",
    "走私国家禁止进出口的货物、物品",
    "走私废物",
    "走私普通货物、物品",
    # ── 隐匿 / 销毁 ──
    "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告",
    # ── 非法 ──
    "非法买卖、运输、携带、持有毒品原植物种子、幼苗",
    "非法制造、买卖、运输、储存危险物质",    # ← previously missing
    "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物",
    "非法制造、出售非法制造的发票",
    "非法制造、销售非法制造的注册商标标识",
    "非法持有、私藏枪支、弹药",
    "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品",  # ← previously missing
    "非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品",
    "非法收购、运输盗伐、滥伐的林木",        # ← previously missing
    "非法猎捕、杀害珍贵、濒危野生动物",
    "非法生产、买卖警用装备",
    "非法生产、销售间谍专用器材",
    "非法转让、倒卖土地使用权",
    "非法采伐、毁坏国家重点保护植物",
    "非法买卖制毒物品",
    "非法侵入住宅",
    "非法出售发票",
    "非法占用农用地",
    "非法吸收公众存款",
    "非法处置查封、扣押、冻结的财产",
    "非法拘禁",
    "非法持有毒品",
    "非法捕捞水产品",
    "非法携带枪支、弹药、管制刀具、危险物品危及公共安全",
    "非法狩猎",
    "非法种植毒品原植物",
    "非法组织卖血",
    "非法经营",
    "非法获取公民个人信息",
    "非法获取国家秘密",
    "非法行医",
    "非法进行节育手术",                      # ← previously missing
    "非法采矿",
    # ── 骗取 ──
    "骗取贷款、票据承兑、金融票证",
    # ── 高利 ──
    "高利转贷",
    # ── 单字 / 短语 ──
    "串通投标",
    "交通肇事",
    "介绍贿赂",
    "以危险方法危害公共安全",
    "传授犯罪方法",
    "传播性病",
    "传播淫秽物品",
    "侮辱",
    "侵占",
    "侵犯著作权",
    "保险诈骗",
    "信用卡诈骗",
    "倒卖文物",
    "倒卖车票、船票",
    "假冒注册商标",
    "冒充军人招摇撞骗",
    "利用影响力受贿",                        # ← previously missing
    "动植物检疫徇私舞弊",
    "劫持船只、汽车",
    "包庇毒品犯罪分子",
    "协助组织卖淫",
    "单位受贿",
    "单位行贿",
    "危险物品肇事",
    "危险驾驶",
    "受贿",
    "合同诈骗",
    "失火",
    "妨害作证",
    "妨害信用卡管理",
    "妨害公务",
    "容留他人吸毒",
    "对单位行贿",
    "对非国家工作人员行贿",
    "寻衅滋事",
    "巨额财产来源不明",
    "帮助毁灭、伪造证据",
    "帮助犯罪分子逃避处罚",
    "开设赌场",
    "强制猥亵、侮辱妇女",
    "强奸",
    "强迫交易",
    "强迫他人吸毒",
    "强迫劳动",
    "强迫卖淫",
    "徇私枉法",
    "徇私舞弊不征、少征税款",
    "徇私舞弊不移交刑事案件",
    "打击报复证人",
    "扰乱无线电通讯管理秩序",
    "投放危险物质",
    "抢劫",
    "抢夺",
    "拐卖妇女、儿童",
    "拐骗儿童",
    "拒不执行判决、裁定",
    "拒不支付劳动报酬",
    "招摇撞骗",
    "招收公务员、学生徇私舞弊",
    "挪用公款",
    "挪用特定款物",
    "挪用资金",
    "提供侵入、非法控制计算机信息系统程序、工具",
    "收买被拐卖的妇女、儿童",
    "放火",
    "故意伤害",
    "故意杀人",
    "故意毁坏财物",
    "敲诈勒索",
    "污染环境",
    "洗钱",
    "滥伐林木",
    "滥用职权",
    "爆炸",                                  # ← previously missing
    "猥亵儿童",
    "玩忽职守",
    "盗伐林木",
    "盗掘古文化遗址、古墓葬",
    "盗窃",
    "破坏广播电视设施、公用电信设施",
    "破坏交通工具",
    "破坏交通设施",
    "破坏易燃易爆设备",
    "破坏生产经营",
    "破坏电力设备",
    "破坏监管秩序",
    "破坏计算机信息系统",
    "票据诈骗",
    "私分国有资产",
    "经济犯",
    "绑架",
    "职务侵占",
    "聚众冲击国家机关",                      # ← previously missing
    "聚众哄抢",
    "聚众扰乱公共场所秩序、交通秩序",
    "聚众扰乱社会秩序",
    "聚众斗殴",
    "脱逃",
    "虐待",
    "虐待被监管人",
    "行贿",
    "诈骗",
    "诬告陷害",
    "诽谤",
    "贪污",
    "贷款诈骗",
    "赌博",
    "过失以危险方法危害公共安全",            # ← previously missing
    "过失投放危险物质",                      # ← previously missing
    "过失损坏广播电视设施、公用电信设施",    # ← previously missing
    "过失损坏武器装备、军事设施、军事通信",
    "过失致人死亡",
    "过失致人重伤",
    "违法发放贷款",
    "逃税",
    "遗弃",
    "重大劳动安全事故",
    "重大责任事故",
    "重婚",
    "金融凭证诈骗",
    "销售假冒注册商标的商品",
    "集资诈骗",
    "非国家工作人员受贿",
]
# fmt: on


_OPTION_SET = None

def extract_predictions(text: str) -> list[str]:
    """Extract predicted accusations from model output.

    Tries structured [罪名]...<eoa> format first (robust against <think> blocks).
    Falls back to option-list scanning on stripped text if the format is absent.
    """
    import re
    global _OPTION_SET
    if _OPTION_SET is None:
        _OPTION_SET = set(OPTION_LIST)

    # Primary: parse the answer block directly
    m = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", text, re.S)
    if m:
        raw = m.group(1).strip()
        labels = [l.strip() for l in raw.split(";") if l.strip()]
        # Normalise: keep only canonical option names, strip "罪" suffix as fallback
        result = []
        for label in labels:
            if label in _OPTION_SET:
                result.append(label)
            elif label.endswith("罪") and label[:-1] in _OPTION_SET:
                result.append(label[:-1])
        if result:
            return result

    # Fallback: scan text with <think> stripped
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return [opt for opt in OPTION_LIST if opt in clean]


def compute_f1(pred_set: set[str], gt_set: set[str]) -> float:
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(
    predictions: list[str],
    ground_truths: list[dict],
) -> dict:
    """Evaluate a list of raw model outputs against ground truth.

    *ground_truths* entries must have an ``accusations`` key (list[str]).
    Returns ``{f1_mean, abstention_rate, total, per_sample}``.
    """
    scores: list[float] = []
    abstentions = 0
    per_sample: list[dict] = []

    for pred_text, gt in zip(predictions, ground_truths):
        pred_set = set(extract_predictions(pred_text))
        gt_set = set(gt["accusations"])
        if not pred_set:
            abstentions += 1
        f1 = compute_f1(pred_set, gt_set)
        scores.append(f1)
        per_sample.append({"pred": list(pred_set), "gt": list(gt_set), "f1": f1})

    return {
        "f1_mean": sum(scores) / max(len(scores), 1),
        "abstention_rate": abstentions / max(len(predictions), 1),
        "total": len(predictions),
        "per_sample": per_sample,
    }
