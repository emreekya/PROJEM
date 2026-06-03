import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

class TestTaskService(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_generate_script_forwards_advanced_prompt_options(self):
        """
        任务生成入口和 WebUI/API 共用 VideoParams。这里验证自动生成文案时，
        高级提示词参数会继续传到 LLM 服务层，避免只在 /scripts 接口生效。
        """
        params = VideoParams(
            video_subject="咖啡",
            video_script="",
            video_language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

        with patch.object(tm.llm, "generate_script", return_value="生成的文案") as generate:
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "生成的文案")
        generate.assert_called_once_with(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

    def test_group_timed_segments_merges_short_subtitle_fragments(self):
        segments = [
            (0.0, 1.0, "Birincisi"),
            (1.0, 3.0, "iletişim becerileri güçlüdür"),
            (3.0, 5.2, "ve insanlarla kolayca konuşabilirler"),
            (5.2, 6.0, "Bu"),
            (6.0, 8.0, "onları sosyal ortamlarda rahat kılar"),
            (8.0, 9.0, "İkincisi"),
            (9.0, 11.5, "problem çözme becerileri gelişmiştir"),
            (11.5, 14.5, "ve yeni yollar deneyebilirler"),
        ]

        grouped = tm._group_timed_segments(
            segments,
            min_duration=4.0,
            target_duration=6.0,
            max_duration=8.0,
        )

        self.assertGreaterEqual(len(grouped), 2)
        self.assertIn("Birincisi iletişim becerileri", grouped[0][2])
        self.assertTrue(any("Bu onları sosyal ortamlarda" in text for _, _, text in grouped))
        self.assertTrue(any("İkincisi problem" in text for _, _, text in grouped))
        self.assertNotIn("Bu", [text for _, _, text in grouped])
        self.assertLessEqual(max(end - start for start, end, _ in grouped), 8.0)

    def test_group_timed_segments_attaches_short_final_scene(self):
        segments = [
            (0.0, 3.0, "Meraklı insanlar yeni bilgiler öğrenir"),
            (3.0, 6.0, "ve farklı deneyimler yaşar"),
            (6.0, 7.0, "Bu"),
            (7.0, 8.0, "yaratıcılığı destekler"),
        ]

        grouped = tm._group_timed_segments(
            segments,
            min_duration=5.0,
            target_duration=6.0,
            max_duration=7.0,
        )

        self.assertTrue(any("Bu yaratıcılığı destekler" in text for _, _, text in grouped))
        self.assertNotIn("Bu", [text for _, _, text in grouped])
        self.assertLessEqual(max(end - start for start, end, _ in grouped), 7.0)

    def test_group_timed_segments_moves_dangling_connector_forward(self):
        segments = [
            (0.0, 4.0, "Yaratıcı çözümler geliştirebilirler"),
            (4.0, 6.0, "Bu"),
            (6.0, 9.0, "onları sanat ve yazarlık alanlarında destekler"),
        ]

        grouped = tm._group_timed_segments(
            segments,
            min_duration=2.0,
            target_duration=4.0,
            max_duration=5.0,
        )

        self.assertEqual(grouped[0][2], "Yaratıcı çözümler geliştirebilirler")
        self.assertTrue(grouped[1][2].startswith("Bu onları sanat"))

    def test_group_timed_segments_splits_long_visual_beats(self):
        grouped = tm._group_timed_segments(
            [
                (
                    0.0,
                    12.0,
                    "Fatih Sultan Mehmet surlari asmak icin ordusunu "
                    "sehrin kapilarina dogru ilerletti",
                ),
            ],
        )

        self.assertGreater(len(grouped), 1)
        self.assertLessEqual(max(end - start for start, end, _ in grouped), 5.5)
    
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials=[]
        for i in range(1, 4):
            video_materials.append(MaterialInfo(
                provider="local",
                url=os.path.join(resources_dir, f"{i}.png"),
                duration=0
            ))

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)
    

if __name__ == "__main__":
    unittest.main()
