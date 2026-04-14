"""
RAG引擎 - 支持在线模型与离线演示模式的客服系统
"""
import os
import re
from typing import Dict, List, Optional

try:
    from langchain_community.chat_models import ChatZhipuAI
    from langchain_community.embeddings import ZhipuAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
except Exception:  # pragma: no cover - 兼容缺依赖或高版本 Python 的导入问题
    ChatZhipuAI = None
    ZhipuAIEmbeddings = None
    Chroma = None
    Document = None
    StrOutputParser = None
    ChatPromptTemplate = None
    RunnablePassthrough = None


DEFAULT_LLM_MODEL = "glm-4"
DEFAULT_EMBEDDING_MODEL = "embedding-3"
SYNONYM_MAP = {
    "退货": "退",
    "退款": "退",
    "换货": "换",
    "面料": "材质",
    "布料": "材质",
    "发货时间": "发货",
    "到货": "物流",
    "邮费": "运费",
}


def _safe_print(message: str) -> None:
    """在不同终端编码下尽量安全地输出日志。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("gbk", errors="replace").decode("gbk"))


class CustomerServiceRAG:
    """支持在线和离线模式的客服 RAG 系统。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        persist_directory: str = "./data/chroma_db",
        temperature: float = 0.7,
        top_k: int = 3,
    ):
        self.api_key = (
            api_key
            or os.getenv("ZHIPUAI_API_KEY", "")
            or os.getenv("ZHIPU_API_KEY", "")
        )
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.top_k = top_k

        self._llm = None
        self._embeddings = None
        self._vectorstore = None
        self._qa_chain = None
        self._merchant_name = "客服"
        self._personality = "热情专业"
        self._documents: List[Dict] = []
        self._mode = "offline"

    def _init_llm(self):
        """初始化智谱AI大语言模型。"""
        if ChatZhipuAI is None:
            raise RuntimeError("当前环境无法加载 ChatZhipuAI，已自动切换为离线模式。")

        if self._llm is None:
            self._llm = ChatZhipuAI(
                api_key=self.api_key,
                model=self.llm_model,
                temperature=self.temperature,
            )
        return self._llm

    def _init_embeddings(self):
        """初始化智谱AI Embedding模型。"""
        if ZhipuAIEmbeddings is None:
            raise RuntimeError("当前环境无法加载 ZhipuAIEmbeddings，已自动切换为离线模式。")

        if self._embeddings is None:
            self._embeddings = ZhipuAIEmbeddings(
                api_key=self.api_key,
                model=self.embedding_model,
            )
        return self._embeddings

    def _keyword_tokens(self, text: str) -> List[str]:
        """提取用于离线匹配的关键词。"""
        normalized = text.lower()
        for source, target in SYNONYM_MAP.items():
            normalized = normalized.replace(source, target)

        tokens = re.findall(r"[\u4e00-\u9fa5]{1,4}|[a-zA-Z0-9]+", normalized)
        chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
        return tokens + chars

    def _score_question(self, question: str, candidate: Dict) -> float:
        """基于关键词重叠和子串命中计算简单相关度。"""
        query_tokens = set(self._keyword_tokens(question))
        candidate_text = f"{candidate.get('question', '')} {candidate.get('answer', '')}"
        candidate_tokens = set(self._keyword_tokens(candidate_text))

        if not query_tokens or not candidate_tokens:
            return 0.0

        overlap = query_tokens & candidate_tokens
        score = len(overlap) / len(query_tokens)

        candidate_question = candidate.get("question", "")
        if question in candidate_question or candidate_question in question:
            score += 0.6

        return score

    def _retrieve_offline(self, question: str) -> List[Dict]:
        """离线模式下检索最相关的问答。"""
        scored = []
        for item in self._documents:
            score = self._score_question(question, item)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in scored[: self.top_k]]

    def _build_offline_answer(self, question: str, sources: List[Dict]) -> str:
        """基于命中的问答生成一个可演示的本地回复。"""
        if not sources:
            return (
                f"您好，这里是{self._merchant_name}。"
                " 目前本地知识库里没有直接匹配到这个问题，"
                "建议您补充商品名称、规格或订单信息，我再为您进一步确认。"
            )

        best = sources[0]
        answer = best.get("answer", "暂时没有合适答案。").strip()
        category = best.get("category", "通用")

        return (
            f"您好，这里是{self._merchant_name}。"
            f" 结合我们现有的{category}知识，{answer}"
        )

    def build_knowledge_base(self, qa_pairs: List[Dict]) -> None:
        """
        构建知识库。

        有可用 API 和依赖时构建向量库，否则自动切换离线演示模式。
        """
        self._documents = list(qa_pairs)

        can_use_online = bool(
            self.api_key and Chroma is not None and Document is not None and ZhipuAIEmbeddings is not None
        )
        if not can_use_online:
            self._mode = "offline"
            _safe_print(f"  [OK] 已加载离线知识库（{len(self._documents)} 条问答）")
            return

        try:
            embeddings = self._init_embeddings()
            documents = []
            for qa in qa_pairs:
                doc_content = f"问题：{qa['question']}\n回答：{qa['answer']}"
                documents.append(
                    Document(
                        page_content=doc_content,
                        metadata={
                            "category": qa.get("category", "通用"),
                            "question": qa["question"],
                            "answer": qa["answer"],
                        },
                    )
                )

            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.persist_directory,
                collection_name="customer_service_kb",
            )
            self._mode = "online"
            _safe_print(f"  [OK] 在线知识库构建完成，已保存至 {self.persist_directory}")
        except Exception as exc:
            self._mode = "offline"
            self._vectorstore = None
            _safe_print(f"  [WARN] 在线知识库构建失败，已切换离线模式：{exc}")
            _safe_print(f"  [OK] 已加载离线知识库（{len(self._documents)} 条问答）")

    def load_knowledge_base(self) -> bool:
        """加载已有的在线知识库。"""
        if not (self.api_key and Chroma is not None and ZhipuAIEmbeddings is not None):
            self._mode = "offline"
            return False

        try:
            embeddings = self._init_embeddings()
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name="customer_service_kb",
            )
            self._mode = "online"
            _safe_print(f"  [OK] 知识库已从 {self.persist_directory} 加载")
            return True
        except Exception as exc:
            self._mode = "offline"
            _safe_print(f"  [WARN] 知识库加载失败，已切换离线模式：{exc}")
            return False

    def setup_qa_chain(
        self,
        merchant_name: str = "客服",
        personality: str = "热情专业",
    ) -> None:
        """配置问答链；离线模式下只保存人设配置。"""
        self._merchant_name = merchant_name
        self._personality = personality

        if self._mode != "online":
            _safe_print(f"  [OK] 离线问答模式已启用（{merchant_name} / {personality}）")
            return

        try:
            llm = self._init_llm()
            retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k},
            )

            template = """你是"{merchant_name}"的智能客服助手，性格{personality}。

请根据以下参考知识回答客户问题。要求：
1. 回答要准确、友好、自然
2. 如果参考知识中有相关信息，优先使用参考知识回答
3. 如果参考知识不足以回答问题，可以结合常识给出合理回答
4. 保持客服语气，适当使用"亲"等亲切称呼

参考知识：
{context}

客户问题：{question}

请给出回答："""

            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(
                    f"[{doc.metadata.get('category', '通用')}] {doc.page_content}"
                    for doc in docs
                )

            self._qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt.partial(
                    merchant_name=self._merchant_name,
                    personality=self._personality,
                )
                | llm
                | StrOutputParser()
            )
            _safe_print(f"  [OK] 在线问答链配置完成（{merchant_name} / {personality}）")
        except Exception as exc:
            self._mode = "offline"
            self._qa_chain = None
            _safe_print(f"  [WARN] 在线问答链配置失败，已切换离线模式：{exc}")
            _safe_print(f"  [OK] 离线问答模式已启用（{merchant_name} / {personality}）")

    def query(self, question: str) -> Dict:
        """查询客服问题。"""
        if self._mode == "online" and self._qa_chain is not None and self._vectorstore is not None:
            retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k},
            )
            relevant_docs = retriever.invoke(question)
            answer = self._qa_chain.invoke(question)

            sources = []
            seen_questions = set()
            for doc in relevant_docs:
                q = doc.metadata.get("question", "")
                if q and q not in seen_questions:
                    sources.append({
                        "category": doc.metadata.get("category", "通用"),
                        "question": q,
                        "answer": doc.metadata.get("answer", ""),
                    })
                    seen_questions.add(q)

            return {
                "answer": answer,
                "sources": sources,
                "mode": "online",
            }

        sources = self._retrieve_offline(question)
        answer = self._build_offline_answer(question, sources)
        return {
            "answer": answer,
            "sources": sources,
            "mode": "offline",
        }
