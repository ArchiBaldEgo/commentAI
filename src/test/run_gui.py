import webbrowser
from threading import Thread
from time import sleep

import uvicorn

from server import app


def run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=9000, reload=False)


def main() -> None:
    t = Thread(target=run_server, daemon=True)
    t.start()

    # даём серверу пару секунд подняться
    sleep(2)
    webbrowser.open("http://127.0.0.1:9000/")

    print("Локальный сервер запущен на http://127.0.0.1:9000/")
    print("Если браузер не открылся автоматически, откройте URL вручную.")

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Остановка.")


if __name__ == "__main__":
    main()
