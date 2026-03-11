FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=9002
ENV LLM_MODEL=gpt-4o
ENV MAX_TOOL_STEPS=15
ENV LLM_TEMPERATURE=0.0

EXPOSE 9002

CMD ["python", "agent.py"]
