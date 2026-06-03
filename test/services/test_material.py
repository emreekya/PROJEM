import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import material


class TestMaterialTlsVerification(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)
        self.original_cloudflare_quota_exhausted = material._cloudflare_quota_exhausted
        self.original_cloudflare_exhausted_accounts = set(
            material._cloudflare_exhausted_accounts
        )
        material._cloudflare_quota_exhausted = False
        material._cloudflare_exhausted_accounts.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)
        material._cloudflare_quota_exhausted = self.original_cloudflare_quota_exhausted
        material._cloudflare_exhausted_accounts.clear()
        material._cloudflare_exhausted_accounts.update(
            self.original_cloudflare_exhausted_accounts
        )

    def test_search_pexels_uses_tls_verification_by_default(self):
        """
        默认路径必须开启 TLS 校验，避免素材 API key 和返回的素材 URL
        在公共网络或不可信代理环境中被中间人攻击截获或篡改。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "duration": 8,
                        "video_files": [
                            {
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response) as get:
            results = material.search_videos_pexels("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_pixabay_allows_explicit_tls_disable_for_proxy(self):
        """
        少数企业代理会使用自签证书。该场景必须显式配置关闭 TLS 校验，
        不能再由代码硬编码默认关闭。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "url": "https://example.com/video.mp4",
                            }
                        },
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response) as get:
            results = material.search_videos_pixabay("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertFalse(get.call_args.kwargs["verify"])

    def test_save_video_uses_tls_verification_by_default(self):
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(content=b"fake-video")

        class FakeVideoFileClip:
            duration = 1
            fps = 24

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "app.services.material.requests.get", return_value=fake_response
            ) as get, patch("app.services.material.VideoFileClip", FakeVideoFileClip):
                video_path = material.save_video(
                    "https://example.com/video.mp4?token=abc", save_dir=temp_dir
                )

            self.assertTrue(os.path.exists(video_path))
            self.assertTrue(get.call_args.kwargs["verify"])

    def test_download_videos_accepts_plain_string_concat_mode(self):
        """
        download_videos 可能被服务层或测试直接传入字符串模式，而不是
        VideoConcatMode 枚举。这里用空搜索词避免真实网络请求，只验证
        字符串 "random" 不会再因为访问 `.value` 抛 AttributeError。
        """
        result = material.download_videos(
            task_id="string-concat-mode",
            search_terms=[],
            video_contact_mode="random",
        )

        self.assertEqual(result, [])

    def test_cloudflare_generic_429_does_not_disable_paid_account(self):
        config.app["cloudflare_account_id"] = "paid-account"
        config.app["cloudflare_api_key"] = "paid-token"
        config.proxy.clear()
        fake_response = SimpleNamespace(
            status_code=429,
            text="Too many requests",
            headers={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "app.services.material.requests.post", return_value=fake_response
            ):
                result = material._generate_one_cloudflare_image(
                    "prompt", 512, 512, 1, temp_dir
                )

        self.assertEqual(result, "")
        self.assertNotIn("paid-account", material._cloudflare_exhausted_accounts)
        self.assertFalse(material._cloudflare_quota_exhausted)

    def test_cloudflare_daily_neuron_limit_disables_free_account(self):
        config.app["cloudflare_account_id"] = "free-account"
        config.app["cloudflare_api_key"] = "free-token"
        config.proxy.clear()
        fake_response = SimpleNamespace(
            status_code=429,
            text="You have exceeded your daily Neurons allocation limit",
            headers={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "app.services.material.requests.post", return_value=fake_response
            ):
                result = material._generate_one_cloudflare_image(
                    "prompt", 512, 512, 1, temp_dir
                )

        self.assertEqual(result, "")
        self.assertIn("free-account", material._cloudflare_exhausted_accounts)
        self.assertTrue(material._cloudflare_quota_exhausted)


if __name__ == "__main__":
    unittest.main()
