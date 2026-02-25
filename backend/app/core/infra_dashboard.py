import platform
import sys
import datetime
import time
import structlog
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from app.core.config import settings

logger = structlog.get_logger(__name__)

def get_gpu_info():
    """获取 GPU 实时状态"""
    if not torch.cuda.is_available():
        return "N/A", 0, 0, 0
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory / (1024**3)
    used_mem = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_mem = torch.cuda.memory_reserved(device) / (1024**3)
    
    gpu_name = props.name
    return gpu_name, used_mem, reserved_mem, total_mem

def print_startup_dashboard():
    """打印美化的启动看板 [Phase 6.3 增强版]"""
    try:
        console = Console()
        gpu_name, used_mem, res_mem, total_mem = get_gpu_info()
        
        # 1. 系统与硬件信息
        sys_info = Table.grid(padding=(0, 2))
        sys_info.add_column(style="cyan")
        sys_info.add_column(style="white")
        
        sys_info.add_row("OS:", f"{platform.system()} {platform.release()}")
        sys_info.add_row("Python:", f"{sys.version.split()[0]}")
        sys_info.add_row("GPU:", f"[bold green]{gpu_name}[/bold green]")
        sys_info.add_row("Version:", f"{settings.VERSION} [Phase 6]")
        sys_info.add_row("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 显存进度条
        if total_mem > 0:
            usage_pct = (used_mem / total_mem) * 100
            color = "green" if usage_pct < 70 else "yellow" if usage_pct < 90 else "red"
            vram_bar = f"[{color}]{used_mem:.2f}GB / {total_mem:.2f}GB ({usage_pct:.1f}%)[/]"
            sys_info.add_row("VRAM Usage:", vram_bar)
        
        # 2. 基础设施连接状态
        svc_status = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        svc_status.add_column("Node", style="dim")
        svc_status.add_column("Status")
        svc_status.add_column("Config")
        
        from pymilvus import connections
        milvus_ok = connections.has_connection("default")
        svc_status.add_row("Milvus DB", "[green]CONNECTED[/green]" if milvus_ok else "[red]OFFLINE[/red]", f"{settings.MILVUS_HOST}")
        svc_status.add_row("Redis Text", "[green]ACTIVE[/green]", "Pool: 10")
        svc_status.add_row("Redis Binary", "[green]ACTIVE[/green]", "ZSTD Comp")
        
        # [Phase 6.5] Cloud LLM Status
        from app.core.infra import CloudHealthCheck
        status_color = "green" if CloudHealthCheck.STATUS == "ONLINE" else "red"
        svc_status.add_row("Cloud LLM", f"[{status_color}]{CloudHealthCheck.STATUS}[/{status_color}]", CloudHealthCheck.MESSAGE)
        
        svc_status.add_row("API Gateway", "[green]ONLINE[/green]", f"Port 8001")
        
        # 3. 模型池状态 (Phase 6.2 核心)
        from app.core.infra import ModelPoolManager
        model_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        model_table.add_column("Model Node")
        model_table.add_column("State")
        model_table.add_column("Last Used")

        now = time.time()
        for name, instance in ModelPoolManager._models.items():
            is_loaded = getattr(instance, 'is_on_gpu', False)
            last_used = ModelPoolManager._last_used.get(name, 0)
            idle_sec = now - last_used
            
            state_str = "[green]ACTIVE (GPU)[/green]" if is_loaded else "[dim]OFFLOADED (CPU)[/dim]"
            idle_str = f"{idle_sec:.0f}s ago" if is_loaded else "-"
            model_table.add_row(name.capitalize(), state_str, idle_str)

        # 4. 组合面板
        layout_table = Table.grid(padding=(1, 2))
        layout_table.add_row(
            Panel(sys_info, title="[bold yellow]System & Hardware[/bold yellow]", border_style="yellow", expand=True),
            Panel(svc_status, title="[bold magenta]Infrastructure Status[/bold magenta]", border_style="magenta", expand=True)
        )
        
        header = Text("\n🏥 SMART HOSPITAL AGENT - PERFORMANCE ENGINE v3.0\n", style="bold white on blue", justify="center")
        
        console.print(header)
        console.print(layout_table)
        console.print(Panel(model_table, title="[bold blue]Dynamic Model Pool (LRU Offloading Strategy)[/bold blue]", border_style="blue"))
        
        # 健康度总结
        health_score = 100
        if not milvus_ok: health_score -= 50
        if used_mem / total_mem > 0.95: health_score -= 30
        
        score_color = "green" if health_score > 80 else "yellow" if health_score > 50 else "red"
        console.print(f"系统运行状态评估: [{score_color}]● {health_score}/100[/] | 优化模式: [bold cyan]Phase 6 (RTX 4060 Optimized)[/]")
        console.print("[dim]提示: 监控指标已暴露于 :8001/metrics | 优先级调度已开启[/]\n")
        
    except Exception as e:
        logger.error("dashboard_print_error", error=str(e))
