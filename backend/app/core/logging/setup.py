import sys
import logging
import structlog
import logging.handlers
import os
from app.core.config import settings

def setup_logging():
    """
    配置全局日志系统 (Global Logging Setup)
    集成 Structlog 和 Standard Logging，输出 JSON 格式日志到文件和控制台。
    """
    # 1. 确定日志级别
    log_level = logging.INFO
    if hasattr(settings, "DEBUG") and settings.DEBUG:
        log_level = logging.DEBUG

    # 2. 获取 Root Logger 并重置 Handlers (防止 Uvicorn 覆盖或重复)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的 handlers (如 Uvicorn 默认的 console handler)
    # 注意：这可能会影响 Uvicorn 的启动日志格式，但在应用层接管是必要的
    # root_logger.handlers = [] 
    # 暂时不暴力清除，以免丢失 Uvicorn 的一些关键输出，而是追加 Handler

    # 3. 配置 Console Handler (标准输出)
    # 使用 Structlog 的 ConsoleRenderer (开发友好) 或 JSONRenderer (生产)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 4. 配置 File Handler (文件输出 - JSON)
    # [Modernization] 使用集中配置的 LOG_DIR
    log_dir = settings.LOG_DIR
    # os.makedirs(log_dir, exist_ok=True) # 已由 settings.LOG_DIR 自动创建
    log_file_path = os.path.join(log_dir, "backend.log")

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)

    # 5. 配置 Structlog Processors
    common_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info, # 异常堆栈
        structlog.processors.UnicodeDecoder(),
    ]
    # structlog 原生日志链路可安全使用 filter_by_level
    processors = [
        structlog.contextvars.merge_contextvars, # 合并 Request ID
        structlog.stdlib.filter_by_level,
        *common_processors,
    ]
    # stdlib foreign_pre_chain 中 logger 可能为 None，不能放 filter_by_level
    foreign_pre_chain = [
        structlog.contextvars.merge_contextvars,
        *common_processors,
    ]

    # JSON Formatter for File
    json_renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    
    # 关键：我们需要一个 ProcessorFormatter 来连接 stdlib logging 和 structlog
    # 这样通过 logging.getLogger().info(...) 调用的日志也能被格式化
    
    # Console Formatter (Keep it readable or JSON)
    # 这里我们让 Console 保持简单，或者也用 JSON
    # console_formatter = structlog.stdlib.ProcessorFormatter(
    #     processor=structlog.dev.ConsoleRenderer(colors=True),
    #     foreign_pre_chain=processors,
    # )
    # console_handler.setFormatter(console_formatter)
    
    # File Formatter (Must be JSON)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=json_renderer,
        foreign_pre_chain=foreign_pre_chain,
    )
    file_handler.setFormatter(file_formatter)

    # 添加 Handlers 到 Root Logger
    # 先检查是否已经存在类似的 handler 防止重复
    has_file_handler = any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in root_logger.handlers)
    if not has_file_handler:
        root_logger.addHandler(file_handler)
        
    # has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in root_logger.handlers)
    # if not has_console_handler:
    #     root_logger.addHandler(console_handler)

    # 6. 配置 Structlog 核心
    structlog.configure(
        processors=processors + [
            # 最后一个 processor 决定最终输出格式。
            # 但我们在 stdlib wrapper 模式下，最后的 rendering 由 Handler 的 Formatter 决定。
            # 这里必须留空或者使用 structlog.stdlib.ProcessorFormatter.wrap_for_formatter
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 验证日志系统
    logger = structlog.get_logger("logging_setup")
    logger.info("global_logging_initialized", log_file=log_file_path)
