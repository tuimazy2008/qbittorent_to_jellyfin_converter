FROM python:3.12-slim

# ffmpeg + VAAPI drivers: Intel (iHD, Broadwell and newer; non-free) and AMD (mesa).
RUN sed -i 's/^Components: main$/Components: main non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg vainfo mesa-va-drivers \
      $( [ "$(dpkg --print-architecture)" = amd64 ] && echo intel-media-va-driver-non-free ) \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV JELLYPREP_CONFIG=/config \
    JELLYPREP_PORT=8765 \
    PYTHONUNBUFFERED=1
VOLUME /config
EXPOSE 8765
CMD ["jellyprep"]
