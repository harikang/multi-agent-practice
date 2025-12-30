# 🎓 Multi-Agent Research Assistant: From Reading to Contributing

## 1. Why Did We Build This? (Motivation)
연구자는 단순한 '독자'가 아니라 '기여(Contribution)를 하는 사람'입니다.  
하나의 논문("Attention is All You Need")을 읽고 새로운 연구를 시작하기 위해서는 다음과 같은 복합적인 과정이 필요합니다:
1.  **이해(Understanding)**: 핵심 아이디어와 구조 파악
2.  **구현(Implementation)**: 밑바닥부터 코드로 구현하며 디테일 확인
3.  **발상(Ideation)**: 기존 한계를 넘어서는 새로운 아이디어 도출
4.  **검증(Due Diligence)**: 내 아이디어가 이미 존재하는지 확인

이 복잡한 **"Research Cycle"**을 가속화하기 위해 Multi-Agent 시스템을 구축했습니다.

---

## 2. User Scenario Breakdown
**"Attention is All You Need"를 기반으로 후속 연구를 진행하고 싶은 상황**

### Phase 1. Deep Understanding (이해)
> **User**: "이 논문을 먼저 완벽하게 이해하고 싶어."

- **🤖 Paper Summary Agent**
    - 논문의 핵심 문제와 해결책을 요약합니다.
    - **Hallucination Free**: 생성된 그림이 아니라, `http_request`로 실제 Ar5iv 논문에 접속해 **진짜 아키텍처 Figure**를 찾아와서 보여줍니다.

### Phase 2. Technical Grounding (구현)
> **User**: "Python으로 Scratch부터 구현해봐야 구조를 뜯어고칠 수 있을 것 같아."

- **🤖 Code Implementation Agent**
    - `python_repl`을 사용하여 실제 실행 가능한 코드를 작성합니다.
    - Multi-Head Attention 같은 특정 모듈 단위로 코드를 분해하여 제공하므로, 유저는 컨트리뷰션 포인트(예: Attention 메커니즘 변경)를 쉽게 찾을 수 있습니다.

### Phase 3. Innovation (발상)
> **User**: "기존 Transformer를 기반으로 더 효율적인 모델을 만들 순 없을까?"

- **🤖 Idea Generation Agent**
    - 논문의 한계점(예: $O(N^2)$ 복잡도)을 분석하여 새로운 아이디어(Linear Attention 등)를 브레인스토밍하여 제안합니다.

### Phase 4. Due Diligence (검증 & 검색)
> **User**: "이 아이디어가 혹시 이미 arXiv에 올라와 있나? 겹치면 안 되는데."

- **🤖 Related Paper Agent (RAG & Search)**
    - 가장 중요한 **중복 검사 및 관련 연구 탐색** 단계입니다.
    - **Smart Workflow**:
        1.  **메모리 확인 (`local_memory_tool` - action: `store`/`retrieve`)**
            - *"내 로컬 DB에 이미 정리해둔 Attention 관련 논문들이 있나?"*
            - SentenceTransformer로 임베딩된 로컬 Faiss DB를 먼저 검색하여 빠르게 답을 찾습니다.
        2.  **외부 검색 (`http_request` / `shell`)**
            - *"DB에 없거나 너무 옛날 정보라면?"*
            - arXiv API를 실시간으로 호출하여 최신 논문을 긁어옵니다.
        3.  **결과 종합**
            - 로컬 지식과 최신 웹 정보를 취합하여 유저에게 "이미 존재하는 연구입니다" 혹은 "새로운 접근입니다"를 알려줍니다.

---

## 3. System Architecture Highlight

### 🛠️ Key Tools & Customization

| Agent | Tool | Type | Description |
|-------|------|------|-------------|
| **Summary** | `extracted_figure` | **Custom Logic** | Ar5iv HTML 파싱 → 실제 논문 그림 Download & Display |
| **Code** | `python_repl` | Built-in | 코드 실행 및 검증 |
| **Related** | **`local_memory_tool`** | **Custom Tool** | **Local RAG System**.<br>- **Embed**: `sentence-transformers`<br>- **Store**: `faiss-cpu`<br>- **Action**: Store/Retrieve |
| **Related** | `http_request` | Built-in | arXiv API 실시간 호출 |

## 4. Conclusion
이 에이전트는 단순한 챗봇이 아니라, **연구자의 사고 과정(Workflow)을 그대로 모사하고 보조**하는 파트너입니다.  
아이디어 발상부터 선행 연구 조사까지, 연구의 병목 구간을 AI가 해결해줍니다.
