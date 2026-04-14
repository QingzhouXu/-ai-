"""
AI客服智能体 - 快速演示脚本
运行此脚本体验完整的客服知识库构建与问答流程。

支持两种模式：
1. 在线模式：设置 ZHIPUAI_API_KEY 或 ZHIPU_API_KEY 后调用智谱模型
2. 离线模式：未配置 API 时，使用本地检索规则完成可演示流程
"""
import os
import sys

# 添加当前目录到Python路径，确保可以导入src模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_processor import ChatDataProcessor
from src.env_loader import load_dotenv
from src.rag_engine import CustomerServiceRAG


def safe_print(message: str) -> None:
    """兼容 Windows 控制台编码的打印方法。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("gbk", errors="replace").decode("gbk"))


def create_sample_data():
    """创建示例聊天记录数据"""
    return [
        # 价格相关
        {"role": "customer", "content": "这个多少钱？"},
        {"role": "merchant", "content": "亲，这款T恤价格是99元，现在店铺活动满200减20哦~"},
        {"role": "customer", "content": "有优惠吗？"},
        {"role": "merchant", "content": "有的亲，关注店铺可以领5元优惠券，新用户还有额外折扣！"},

        # 物流相关
        {"role": "customer", "content": "什么时候发货？"},
        {"role": "merchant", "content": "亲，16:00前下单当天发货，之后次日发货，默认发中通快递~"},
        {"role": "customer", "content": "运费怎么算？"},
        {"role": "merchant", "content": "全场满99包邮，不满的话运费6元，偏远地区需要补差价哦~"},

        # 售后相关
        {"role": "customer", "content": "不喜欢可以退吗？"},
        {"role": "merchant", "content": "支持7天无理由退换，只要不影响二次销售就可以，运费险也送您~"},
        {"role": "customer", "content": "质量有问题怎么办？"},
        {"role": "merchant", "content": "质量问题我们包退包换，您直接申请售后，我们承担运费~"},

        # 产品相关
        {"role": "customer", "content": "这是什么材质的？"},
        {"role": "merchant", "content": "亲，这款是100%纯棉的，亲肤透气，夏天穿很舒适~"},
        {"role": "customer", "content": "尺码准吗？"},
        {"role": "merchant", "content": "尺码是标准码，建议按照平时尺码选，不确定可以问客服推荐~"},

        # 库存相关
        {"role": "customer", "content": "还有货吗？"},
        {"role": "merchant", "content": "目前白色和黑色还有现货，其他颜色需要3-5天补货~"},
        {"role": "customer", "content": "能预定吗？"},
        {"role": "merchant", "content": "可以预定的亲，拍下备注预定，到货后第一时间给您发货~"},
    ]


def main():
    """主流程演示"""
    dotenv_loaded = load_dotenv()

    safe_print("=" * 60)
    safe_print("AI客服智能体 - 可运行演示")
    safe_print("=" * 60)

    # 步骤1: 数据清洗
    safe_print("\n[步骤1] 数据清洗与处理")
    safe_print("-" * 60)

    processor = ChatDataProcessor()
    chat_records = create_sample_data()

    safe_print(f"原始聊天记录：{len(chat_records)} 条消息")
    qa_pairs = processor.extract_qa_pairs(chat_records)
    safe_print(f"提取有效问答对：{len(qa_pairs)} 条")

    # 显示处理后的数据
    safe_print("\n处理后的问答对：")
    for i, qa in enumerate(qa_pairs, 1):
        safe_print(f"\n{i}. [{qa['category']}] {qa['question']}")
        safe_print(f"   -> {qa['answer']}")

    # 保存数据
    os.makedirs("data", exist_ok=True)
    processor.save_to_json(qa_pairs, "data/qa_pairs.json")

    # 步骤2: 初始化 RAG
    safe_print("\n" + "=" * 60)
    safe_print("[步骤2] RAG系统初始化")
    safe_print("-" * 60)

    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if api_key:
        safe_print("[OK] 检测到智谱 API 密钥，将优先尝试在线模式")
    else:
        safe_print("[INFO] 未检测到智谱 API 密钥，将使用离线演示模式")
        if dotenv_loaded:
            safe_print("       已读取 .env，但其中未找到可用的智谱 API 密钥")
        else:
            safe_print("       当前未找到 .env 文件，也未设置系统环境变量")
        safe_print("       如需在线模式，可在 .env 中填写 ZHIPUAI_API_KEY 或 ZHIPU_API_KEY")

    # 步骤3: 构建 RAG 系统并测试
    rag = CustomerServiceRAG(api_key=api_key, persist_directory="./data/chroma_db")

    try:
        rag.build_knowledge_base(qa_pairs)
        rag.setup_qa_chain(merchant_name="时尚T恤店", personality="热情活泼")

        # 步骤4: 测试查询
        safe_print("\n" + "=" * 60)
        safe_print("[步骤3] 智能客服测试")
        safe_print("-" * 60)

        test_questions = [
            "这件T恤多少钱？",
            "今天下单什么时候能到？",
            "如果不喜欢可以退货吗？",
            "是什么面料的？",
        ]

        for question in test_questions:
            safe_print(f"\n客户：{question}")
            result = rag.query(question)
            safe_print(f"模式：{result.get('mode', 'unknown')}")
            safe_print(f"客服：{result['answer']}")
            safe_print(f"参考知识：{len(result['sources'])} 条")
            for source in result['sources']:
                safe_print(f"  - [{source['category']}] {source['question']}")

        safe_print("\n" + "=" * 60)
        safe_print("[OK] 演示完成")
        safe_print("=" * 60)

    except Exception as e:
        safe_print(f"\n[ERROR] 运行出错：{e}")
        safe_print("\n可能的原因：")
        safe_print("  1. 依赖包未正确安装（运行 pip install -r requirements.txt）")
        safe_print("  2. 在线模式下 API 密钥无效、余额不足或网络不可用")
        safe_print("  3. 当前 Python / LangChain 版本兼容性存在问题")


if __name__ == "__main__":
    main()
