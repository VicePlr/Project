import base64
import cgi
import html
import os
import shutil
import sys
import tempfile
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enroll_user import enroll_user, resolve_path


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.dirname(__file__)
USERS_CSV_PATH = resolve_path("data", "users", "users.csv")
FACES_DIR_PATH = resolve_path("data", "users", "faces")
TEMPLATE_PATH = os.path.join(SRC_ROOT, "templates", "guest_registration.html")
STYLESHEET_PATH = os.path.join(SRC_ROOT, "static", "styles.css")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as input_file:
        return input_file.read()


def safe_filename(filename: str) -> str:
    basename = os.path.basename(filename or "")
    sanitized = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in basename
    )
    return sanitized or "upload.bin"


def save_uploaded_file(field_item, destination_dir: str) -> str:
    filename = safe_filename(field_item.filename)
    destination_path = os.path.join(destination_dir, filename)
    with open(destination_path, "wb") as upload_file:
        shutil.copyfileobj(field_item.file, upload_file)
    return destination_path


def guess_extension_from_data_url(data_url_header: str) -> str:
    mime_type = data_url_header.split(";", 1)[0].split(":", 1)[-1].lower()
    return {
        "video/webm": ".webm",
        "video/mp4": ".mp4",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }.get(mime_type, ".webm")


def save_recorded_media(data_url: str, destination_dir: str) -> str:
    if not data_url:
        raise ValueError("Dữ liệu khuôn mặt đã quay đang trống.")
    if "," not in data_url or ";base64" not in data_url:
        raise ValueError("Dữ liệu khuôn mặt đã quay không hợp lệ.")

    header, encoded_data = data_url.split(",", 1)
    extension = guess_extension_from_data_url(header)
    destination_path = os.path.join(destination_dir, f"recorded_face{extension}")

    with open(destination_path, "wb") as output_file:
        output_file.write(base64.b64decode(encoded_data))
    return destination_path


def render_page(message=None, message_tone="success"):
    template_html = read_text_file(TEMPLATE_PATH)
    message_html = ""
    if message:
        message_html = f"""
        <div class="banner banner-{html.escape(message_tone)}">
          <p>{html.escape(message)}</p>
        </div>
        """
    return template_html.replace("__MESSAGE_HTML__", message_html)


class VehicleAccessPortalHandler(BaseHTTPRequestHandler):
    server_version = "VehicleAccessPortal/1.0"

    def log_message(self, format, *args):
        return

    def _send_response_body(
        self,
        body,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
        encoding: str = "utf-8",
    ):
        if isinstance(body, str):
            encoded = body.encode(encoding)
        else:
            encoded = body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html_content: str, status: int = HTTPStatus.OK):
        self._send_response_body(
            html_content,
            content_type="text/html; charset=utf-8",
            status=status,
        )

    def _send_css(self, css_content: str, status: int = HTTPStatus.OK):
        self._send_response_body(
            css_content,
            content_type="text/css; charset=utf-8",
            status=status,
        )

    def _render(self, *, message=None, message_tone="success", status=HTTPStatus.OK):
        self._send_html(
            render_page(
                message=message,
                message_tone=message_tone,
            ),
            status=status,
        )

    def _parse_form(self):
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
        }
        content_length = self.headers.get("Content-Length")
        if content_length:
            environ["CONTENT_LENGTH"] = content_length
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
        )

    @staticmethod
    def _get_optional_file_field(form, field_name: str):
        if field_name not in form:
            return None
        field = form[field_name]
        if not getattr(field, "filename", ""):
            return None
        return field

    def do_GET(self):
        if self.path == "/":
            self._render()
            return
        if self.path == "/styles.css":
            self._send_css(read_text_file(STYLESHEET_PATH))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Không tìm thấy trang.")

    def do_POST(self):
        try:
            if self.path == "/register":
                self._handle_register()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Không tìm thấy trang.")
        except Exception as exc:
            traceback.print_exc()
            self._render(
                message=str(exc),
                message_tone="error",
                status=HTTPStatus.BAD_REQUEST,
            )

    def _handle_register(self):
        form = self._parse_form()
        plate = form.getfirst("plate", "").strip()
        face_media = self._get_optional_file_field(form, "face_media")
        recorded_face_data = form.getfirst("recorded_face_data", "").strip()

        if not plate:
            raise ValueError("Vui lòng nhập biển số xe.")
        if face_media is None and not recorded_face_data:
            raise ValueError("Vui lòng tải ảnh/video khuôn mặt hoặc quay video trực tiếp.")

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            if face_media is not None:
                temp_source_path = save_uploaded_file(face_media, temp_dir)
            else:
                temp_source_path = save_recorded_media(recorded_face_data, temp_dir)

            is_video = os.path.splitext(temp_source_path)[1].lower() in VIDEO_EXTENSIONS
            enrollment_result = enroll_user(
                "",
                plate,
                temp_source_path,
                users_csv=USERS_CSV_PATH,
                faces_dir=FACES_DIR_PATH,
                is_video=is_video,
                guest_mode=True,
            )

        registered_user_id = enrollment_result["user_id"]
        self._render(
            message=f"Đăng ký khách thành công. Mã khách là: {registered_user_id}",
            message_tone="success",
        )


def create_server(host: str = "localhost", port: int = 8000):
    return ThreadingHTTPServer((host, port), VehicleAccessPortalHandler)


def main():
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", "8000"))
    server = create_server(host, port)
    print(f"Cổng đăng ký khách đang chạy tại http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
