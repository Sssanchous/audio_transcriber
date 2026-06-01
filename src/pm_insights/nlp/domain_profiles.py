from __future__ import annotations


PROFILES = {
    "project_meeting": {
        "max_block_words": 100,
        "task_accept_threshold": 0.75,
        "task_review_threshold": 0.45,
        "strict_tasks": False,
        "aspect_profile": "project",
    },
    "technical_research": {
        "max_block_words": 140,
        "task_accept_threshold": 0.8,
        "task_review_threshold": 0.5,
        "strict_tasks": True,
        "aspect_profile": "technical",
    },
    "education_consultation": {
        "max_block_words": 120,
        "task_accept_threshold": 0.8,
        "task_review_threshold": 0.5,
        "strict_tasks": True,
        "aspect_profile": "education",
    },
    "mixed": {
        "max_block_words": 120,
        "task_accept_threshold": 0.78,
        "task_review_threshold": 0.5,
        "strict_tasks": True,
        "aspect_profile": "mixed",
    },
    "general_discussion": {
        "max_block_words": 100,
        "task_accept_threshold": 0.8,
        "task_review_threshold": 0.5,
        "strict_tasks": True,
        "aspect_profile": "general",
    },
}


def get_domain_profile(meeting_type_label: str | None) -> dict:
    return dict(PROFILES.get(meeting_type_label or "", PROFILES["general_discussion"]))
