"""邮件发送模块 — SMTP 发送信号日报到指定邮箱

参照 DataGetter/data_generate_tester.ipynb 中的邮件发送逻辑。
配置区集中管理 SMTP 参数。
"""
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional


# ─── 配置区 ───
SMTP_SERVER = "smtp.qq.com" # 实际使用时请替换为自己的 SMTP 服务器地址
SMTP_PORT = 587
SENDER = "your_email@qq.com" # 实际使用时请替换为自己的邮箱
AUTH_CODE = "your_auth_code" # 实际使用时请替换为自己的授权码
DEFAULT_RECEIVERS = ["your_email@qq.com"] # 实际使用时请替换为自己的邮箱
# ─────────────


def send_report_email(
    html_path: str,
    receivers: Optional[List[str]] = None,
    subject_prefix: str = "信号日报",
):
    """发送 HTML 日报到指定邮箱

    Args:
        html_path: HTML 报告文件路径
        receivers: 收件人列表 (默认 DEFAULT_RECEIVERS)
        subject_prefix: 邮件主题前缀
    """
    if receivers is None:
        receivers = DEFAULT_RECEIVERS

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"报告文件不存在: {html_path}")

    today = datetime.now().strftime("%Y-%m-%d")

    msg = MIMEMultipart()
    msg["From"] = SENDER
    msg["To"] = ",".join(receivers)
    msg["Subject"] = f"{subject_prefix} - {today}"

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER, AUTH_CODE)
            server.sendmail(SENDER, receivers, msg.as_string())
        print(f"邮件已发送至: {', '.join(receivers)}", flush=True)
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}", flush=True)
        return False
