# 使用 Python 3.9 slim 版本作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装必要的包
RUN apt-get update && apt-get install -y 

# 复制项目文件到工作目录
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 对外暴露端口，如果你的 app 使用不同的端口，需要修改这里
EXPOSE 5001

# 指定容器启动时的默认命令
CMD ["python", "app.py"]

