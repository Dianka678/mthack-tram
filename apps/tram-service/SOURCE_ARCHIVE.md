# Исходники в рабочей ветке

`tram-service-code.zip` временно содержит дерево исходников (`app/`, `scripts/`, `tests/`, `data/*.example.*`) без конкурсного CSV и секретов. В каталоге `apps/tram-service` распакуйте его командой `unzip -o tram-service-code.zip`. Для Windows PowerShell: `Expand-Archive .\tram-service-code.zip -DestinationPath . -Force`.

Затем поместите полученный от владельца конкурсный файл в `data/forecast.csv`; не коммитьте его в публичные ветки. Создайте пароль `python3 scripts/create_auth.py`, запустите `docker compose up --build -d`, откройте `http://127.0.0.1:8080/`. План DS и контракт событий: `../../docs/DS_REQUIREMENTS.md`.

Это промежуточная упаковка для передачи, **не merge-ready PR**: перед слиянием разложить исходники в Git как обычные файлы, прогнать тесты и проверить правила приватности. Автоматический численный эффект дождя/ремонта требует поставки DS-модели.
