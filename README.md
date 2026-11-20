# jellyprep

Prepares finished qBittorrent downloads for a Jellyfin library:

- picks up completed torrents from the categories you choose;
- identifies movies / TV episodes (guessit + extra rules for Russian release names + TMDB);
- lays them out with Jellyfin naming (`Title (Year) [tmdbid-N]/…`, `Season 01/Show S01E01.mkv`) using hardlinks;
- asks you to confirm anything it is unsure about;
- removes the original torrent once it has seeded enough (ratio **or** days), only after checking the library copy.

- re-encodes what doesn't play everywhere (4K, HDR, HEVC, huge bitrates) to 1080p H.264 on the GPU (VAAPI: Intel
  Quick Sync, AMD), with proper HDR → SDR tone mapping, during a night window;
- adds a "voice boost" stereo track (louder dialogue, tamed explosions, normalized loudness) and drops audio and
  subtitle languages you don't need;
- moves files from your old library folders through the same pipeline ("Import" page).

Planned: an optional local LLM (Ollama, LM Studio) for hard-to-parse names, English UI, NVIDIA NVENC.
See [SPEC.md](SPEC.md) (Russian).

## Run

```sh
docker build -t jellyprep .
# edit docker-compose.example.yml (media path, LAN address), then:
docker compose -f docker-compose.example.yml up -d
```

Open `http://<lan-ip>:8765`. The first-run wizard walks through every option. It starts in **dry-run** mode: it shows
what it would do and changes nothing until you switch dry-run off.

The UI has no login: publish the port on your LAN address only.

GPU encoding needs `/dev/dri` passed into the container and the host's `render` group id in `group_add`
(see the compose example). The wizard has a "check GPU" button.

## Develop

```sh
uv sync
uv run pytest
JELLYPREP_CONFIG=./config uv run jellyprep
```
