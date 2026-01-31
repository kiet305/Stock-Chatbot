from typing import List, Dict, Any


class AnswerSynthesizerTool:
    def synthesize_final_answer(self, agent_outputs: List[Dict[str, Any]]) -> str:
        """
        Tổng hợp kết quả trả về từ nhiều agent thành 1 câu trả lời cuối cùng.

        agent_outputs: [
          {"agent": "CustomerServiceAgent", "content": "..."},
          {"agent": "GoogleSearchAgent", "content": "..."}
        ]
        """

        if not agent_outputs:
            return "Mình chưa nhận được dữ liệu từ các agent để tổng hợp."

        merged_lines = []
        for item in agent_outputs:
            agent_name = item.get("agent", "UnknownAgent")
            content = str(item.get("content", "")).strip()

            if content:
                merged_lines.append(f"- ({agent_name}) {content}")

        if not merged_lines:
            return "Mình chưa nhận được nội dung hợp lệ từ các agent để tổng hợp."

        final_answer = (
            "Dưới đây là thông tin mình đã tổng hợp từ các agent:\n\n"
            + "\n".join(merged_lines)
            + "\n\nNếu bạn muốn, mình có thể tóm tắt ngắn gọn hoặc chuyển thành checklist."
        )

        return final_answer