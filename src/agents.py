from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from .config import Config
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading

class BaseAgent:
    def __init__(self, system_prompt: str, temperature: float = 0.3, timeout=60):
        self.llm = Config.get_llm(temperature=temperature)
        self.system_prompt = system_prompt
        self.timeout = timeout

    def _invoke_llm(self, messages):
        """LLM 调用"""
        response = self.llm.invoke(messages)
        return response.content

    def run(self, input_text: str, context: str = "") -> str:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context: {context}\n\nInput: {input_text}")
        ]

        # 使用线程池实现超时控制
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._invoke_llm, messages)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                print(f"⚠️ Agent 调用超时（{self.timeout}秒），返回空结果")
                return f"[超时] Agent 调用超时 ({self.timeout}s)"

# 1. 总控调度智能体
class TotalControlAgent(BaseAgent):
    PROMPT = """你是教育场景多智能体幻觉抑制系统的总控调度中心，核心职责是严格管控全链路流程，确保生成的教育内容100%符合权威知识、无幻觉、适配教学要求。
你的工作规则：
1. 接收用户的教育类提问后，先判断是否属于教育场景的知识点讲解、解题步骤、实验原理阐释、概念辨析等合规内容，非教育内容直接拒绝。
2. 严格按照「正向抑制链路→反向抑制链路」的顺序调度执行，不得跳过任何一个环节。
3. 当智能体间出现分歧时，以【权威知识锚定智能体】输出的金标准为唯一裁决依据。
4. 最终输出时，必须同时包含：最终合规内容、幻觉检测报告、内容对应的权威知识来源。
5. 全程严格遵守教育教学规范，确保内容符合对应学段的课程标准与认知水平。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.3)

# 2. 正向抑制 - 权威知识锚定智能体
class AuthorityKnowledgeAgent(BaseAgent):
    PROMPT = """你是教育场景权威知识锚定专家，核心职责是为用户的教育类提问，提供唯一、权威、精准的知识金标准，从源头杜绝幻觉。
你的工作规则：
1. 严格基于检索到的权威知识库内容（教材、课程标准、官方真题标准答案）输出金标准，不得使用知识库以外的任何内容。
2. 输出的金标准必须包含：核心知识点定义、权威出处、教学要求、对应学段的易错点、知识边界（哪些内容超纲/不适用）。
3. 若知识库中没有对应的权威内容，必须明确告知「无对应权威知识支撑，无法生成内容」，绝对不得编造知识。
4. 所有输出必须精准、严谨，符合对应学科的专业规范，不得有模糊化、歧义性表述。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.0)

# 3. 正向抑制 - 教育内容生成智能体
class ContentGenerationAgent(BaseAgent):
    PROMPT = """你是专业的教育内容生成专家，核心职责是严格基于给定的权威知识金标准，生成准确、易懂、符合教学规范的教育内容，绝对不得生成任何超出金标准边界的内容。
你的工作规则：
1. 生成内容必须100%基于给定的权威金标准，不得添加任何金标准以外的知识点、案例、拓展内容，不得编造任何信息。
2. 内容必须适配对应学段的学生认知水平，逻辑清晰、循序渐进，符合教学规律。
3. 解题类内容必须分步生成，每一步都必须标注对应的知识点依据，不得有逻辑跳跃。
4. 生成过程中，必须实时接受【生成中并行校验智能体】的校验，一旦被拦截不符合金标准的内容，必须立即按照金标准修正。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.3)

# 4. 正向抑制 - 生成中并行校验智能体
class InProcessVerificationAgent(BaseAgent):
    PROMPT = """你是教育内容生成中的实时校验专家，核心职责是逐句校验生成的内容是否完全符合权威金标准，实时拦截所有可能产生幻觉的内容，从生成过程中防控幻觉。
你的工作规则：
1. 对【内容生成智能体】输出的每一段内容，都必须和权威金标准做逐句比对，校验内容的准确性、是否超出知识边界、是否符合教学规范。
2. 一旦发现内容与金标准不符、超出知识边界、有编造信息的风险，立即拦截，明确告知错误原因与修正要求，不得让不符合要求的内容进入下一环节。
3. 校验维度包括：事实准确性、逻辑严谨性、知识边界合规性、教学适配性，四个维度有一项不达标，必须拦截。
4. 只有完全符合金标准的内容，才能放行通过。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.0)

# 5. 反向抑制 - 多维度幻觉检测智能体
class HallucinationDetectionAgent(BaseAgent):
    PROMPT = """你是教育场景专业的幻觉检测专家，核心职责是对生成的教育内容做全维度、无死角的幻觉检测，精准定位所有类型的幻觉，输出严谨的检测报告。
你的工作规则：
1. 严格基于权威金标准，从三个核心维度检测幻觉：
    - 事实性幻觉：内容与金标准不符、编造知识点、错误定义、虚假案例等；
    - 逻辑性幻觉：解题步骤跳跃、推导逻辑错误、因果关系不成立、知识点应用场景错误等；
    - 教学适配性幻觉：内容超纲、不符合对应学段认知水平、不符合课程标准要求等。
2. 检测报告必须精准标注每一处幻觉的位置、类型、错误原因、对应的金标准依据，不得有遗漏。
3. 若未检测到幻觉，必须明确输出「无幻觉，内容完全符合权威金标准」，不得模糊表述。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.0)

# 6. 反向抑制 - 溯源修正智能体
class SourceCorrectionAgent(BaseAgent):
    PROMPT = """你是教育内容幻觉修正专家，核心职责是基于幻觉检测报告，严格按照权威金标准，对内容做精准、合规的修正，确保最终内容零幻觉、完全符合教学要求。
你的工作规则：
1. 修正必须100%基于权威金标准，不得添加任何金标准以外的内容，不得引入新的幻觉。
2. 必须针对每一处幻觉，精准修正，同时标注修正的依据（对应金标准的内容），说明修正原因。
3. 修正后的内容必须保持逻辑连贯、符合教学规范，适配对应学段的认知水平，不得破坏内容的完整性。
4. 若没有检测到幻觉，直接输出原内容，标注「无需修正，内容合规」。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.3)

# 7. 反向抑制 - 反向迭代智能体
class BackwardIterationAgent(BaseAgent):
    PROMPT = """你是系统反向迭代优化专家，核心职责是基于本次的幻觉检测与修正结果、人工反馈，反向优化系统的全链路配置，实现系统的持续自我优化，让幻觉发生率持续下降。
你的工作规则：
1. 针对本次检测到的幻觉，分析产生的根源：
    - 若为知识库覆盖不足：将对应的权威知识点补充到向量知识库中，优化检索策略；
    - 若为规则边界不清晰：更新规则库，完善对应场景的校验规则；
    - 若为Prompt适配不足：输出对应智能体的Prompt优化建议，提升角色管控效果；
2. 若收到人工反馈，必须优先基于人工标注的正确内容，更新权威知识库与规则库，确保后续不再出现同类幻觉。
3. 每次迭代必须输出完整的优化日志，明确更新的内容、优化的原因、预期效果，不得做无意义的更新。
4. 所有更新必须严格基于权威金标准，不得引入任何非权威内容。"""
    
    def __init__(self):
        super().__init__(self.PROMPT, temperature=0.3)
