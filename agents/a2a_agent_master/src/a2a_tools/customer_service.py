import httpx
from uuid import uuid4

from a2a.client import A2AClient
from a2a.types import (
    SendMessageRequest,
    MessageSendParams,
    Message,
    Role,
    Part,
    TextPart,
    TaskState,
)

from google.adk.tools import ToolContext

class CustomerServiceAgent:
    def __init__(self, agent_url: str):
        self.agent_url = agent_url

    async def invoke_customer_service_agent_via_a2a(
        self, user_message: str, tool_context: ToolContext
    ):
        """
        A helpful customer service assistant help delegate tasks to other agents.
        For searching related to stocks, finance, economics, and company information, also company metrics liek roa, roe, bvps, eps, nim, ros.

        Args:
            user_message: A question related to company or searching infomation about company or stocks, finance, economics
        """
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message=Message(
                    contextId=tool_context._invocation_context.session.id,
                    taskId=tool_context.state.get('task_id'),
                    messageId=str(uuid4()),
                    role=Role.user,
                    parts=[Part(TextPart(text=user_message))],
                )
            ),
        )

        self._update_status(tool_context)

        try:
            async with httpx.AsyncClient(timeout=60) as httpx_client:
                client = await A2AClient.get_client_from_agent_card_url(
                    httpx_client, self.agent_url
                )
                return await client.send_message(request)
        except Exception as e:
            print(f'An error occurred: {e}')

    def _update_status(self, tool_context: ToolContext):
        try:
            task_updater = tool_context._invocation_context.run_config.current_task_updater
            task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    [Part(TextPart(text='Waiting for sql agent company agent to greet you back...'))]
                ),
            )
        except Exception as e:
            print(f"Error updating status: {e} ; You might be using adk-web where the task updater is not available")