import time
from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat")

# 设置 OpenTelemetry Tracer
trace.set_tracer_provider(TracerProvider(resource=Resource.create({SERVICE_NAME: "LangChainService"})))
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter()
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# 设置 OpenTelemetry Meter
meter_provider = MeterProvider(resource=Resource.create({SERVICE_NAME: "LangChainService"}))
meter = meter_provider.get_meter("langchain_metrics", version="0.1")
metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())

# 创建度量
requests_counter = meter.create_counter(
    name="requests",
    description="Number of requests",
    unit="1",
)
requests_duration = meter.create_histogram(
    name="requests_duration",
    description="Duration of requests",
    unit="ms",
)


class MonitoringCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain start")
        self.llm_span = tracer.start_span("Chain call")
        self.llm_start_time = time.time()

    def on_chain_end(self, outputs, **kwargs):
        print("Chain end")
        self.llm_span.end()
        duration = (time.time() - self.llm_start_time) * 1000
        requests_duration.record(duration, {"operation": "chain"})
        requests_counter.add(1, {"operation": "chain", "status": "success"})

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM start")
        self.retriever_span = tracer.start_span("LLM call")
        self.retriever_start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        print("LLM end")
        self.retriever_span.end()
        duration = (time.time() - self.retriever_start_time) * 1000
        requests_duration.record(duration, {"operation": "llm"})
        requests_counter.add(1, {"operation": "llm", "status": "success"})


if __name__ == "__main__":
    callbacks = [MonitoringCallbackHandler()]
    prompt = ChatPromptTemplate.from_template("给做{product}的公司，取一个名字，不超过5个字")
    chain = prompt | llm
    output = chain.invoke({"product": "杯子"}, config={"callbacks": callbacks})
    print(output)
