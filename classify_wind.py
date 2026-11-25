import csv
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("wind_dataset.csv")
PREDICTIONS_PATH = Path(__file__).with_name("classification_predictions.csv")
SCORES_PATH = Path(__file__).with_name("classification_scores.csv")
THRESHOLD_MS = 4.0
ONE_HOUR = timedelta(hours=1)
FEATURE_COLUMNS = (
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_ago_ms",
    "wind_speed_2h_ago_ms",
)


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    required_columns = {
        "prediction_time",
        "target_time",
        "target_wind_speed_ms",
        *FEATURE_COLUMNS,
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required_columns))}")

    previous_prediction_time = None
    previous_target_time = None
    target_time_gaps: list[tuple[datetime, datetime, timedelta]] = []
    for row_number, row in enumerate(rows, start=2):
        prediction_time = datetime.strptime(row["prediction_time"], "%Y-%m-%d %H:%M:%S")
        target_time = datetime.strptime(row["target_time"], "%Y-%m-%d %H:%M:%S")

        if target_time - prediction_time != ONE_HOUR:
            raise ValueError(
                f"Row {row_number}: prediction_time and target_time are not exactly 1 hour apart"
            )
        if previous_prediction_time is not None and prediction_time <= previous_prediction_time:
            raise ValueError(f"Row {row_number}: prediction_time is not in increasing order")
        if previous_target_time is not None:
            target_gap = target_time - previous_target_time
            if target_gap <= timedelta(0):
                raise ValueError(f"Row {row_number}: target_time is not in increasing order")
            if target_gap != ONE_HOUR:
                target_time_gaps.append((previous_target_time, target_time, target_gap))

        row["_parsed_prediction_time"] = prediction_time
        row["_parsed_target_time"] = target_time
        row["_label"] = int(float(row["target_wind_speed_ms"]) >= THRESHOLD_MS)
        previous_prediction_time = prediction_time
        previous_target_time = target_time

    train_start = datetime(2023, 1, 1)
    evaluation_start = datetime(2024, 12, 1)
    evaluation_end = datetime(2025, 1, 1)

    train_rows = [
        row
        for row in rows
        if train_start <= row["_parsed_target_time"] < evaluation_start
    ]
    evaluation_rows = [
        row
        for row in rows
        if evaluation_start <= row["_parsed_target_time"] < evaluation_end
    ]
    if not train_rows:
        raise ValueError("No training rows found in the requested date range")
    if not evaluation_rows:
        raise ValueError("No evaluation rows found in December 2024")

    def features(selected_rows: list[dict[str, str]]) -> list[list[float]]:
        return [
            [float(row[column]) for column in FEATURE_COLUMNS]
            for row in selected_rows
        ]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=None),
    )
    model.fit(features(train_rows), [row["_label"] for row in train_rows])
    strong_wind_probabilities = model.predict_proba(features(evaluation_rows))[
        :, list(model.classes_).index(1)
    ]
    actual_labels = [int(row["_label"]) for row in evaluation_rows]
    method_predictions = {
        "항상 0": [0] * len(evaluation_rows),
        "현재 풍속 >= 4.0 m/s": [
            int(float(row["wind_speed_ms"]) >= THRESHOLD_MS)
            for row in evaluation_rows
        ],
        "로지스틱 회귀 (확률 >= 0.50)": [
            int(probability >= 0.50)
            for probability in strong_wind_probabilities
        ],
        "로지스틱 회귀 (확률 >= 0.20)": [
            int(probability >= 0.20)
            for probability in strong_wind_probabilities
        ],
    }

    def calculate_scores(
        predictions: list[int],
    ) -> dict[str, int | float | str]:
        hits = sum(
            actual == 1 and predicted == 1
            for actual, predicted in zip(actual_labels, predictions)
        )
        misses = sum(
            actual == 1 and predicted == 0
            for actual, predicted in zip(actual_labels, predictions)
        )
        false_alarms = sum(
            actual == 0 and predicted == 1
            for actual, predicted in zip(actual_labels, predictions)
        )
        correct_rejections = sum(
            actual == 0 and predicted == 0
            for actual, predicted in zip(actual_labels, predictions)
        )
        predicted_positive_count = hits + false_alarms
        actual_positive_count = sum(actual_labels)
        return {
            "evaluation_rows": len(evaluation_rows),
            "actual_strong_wind_count": actual_positive_count,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_rejections": correct_rejections,
            "accuracy": (hits + correct_rejections) / len(evaluation_rows),
            "strong_wind_precision": (
                hits / predicted_positive_count
                if predicted_positive_count
                else "해당없음"
            ),
            "strong_wind_recall": (
                hits / actual_positive_count
                if actual_positive_count
                else "해당없음"
            ),
        }

    scores = {
        method: calculate_scores(predictions)
        for method, predictions in method_predictions.items()
    }
    if len({score["evaluation_rows"] for score in scores.values()}) != 1:
        raise RuntimeError("Methods were not evaluated on the same number of rows")
    if len({score["actual_strong_wind_count"] for score in scores.values()}) != 1:
        raise RuntimeError("Methods were not evaluated against the same actual labels")

    def print_summary(name: str, selected_rows: list[dict[str, str]]) -> None:
        positive_count = sum(int(row["_label"]) for row in selected_rows)
        negative_count = len(selected_rows) - positive_count
        print(
            f"{name}: 전체 {len(selected_rows)}행, "
            f"0 {negative_count}개, 1 {positive_count}개"
        )

    print("시간 기준: KST (CSV의 시간값을 별도 변환 없이 사용)")
    print("검증: 모든 행의 prediction_time → target_time = 정확히 1시간")
    print("검증: prediction_time과 target_time 모두 오름차순")
    print(f"행 사이 1시간 간격이 아닌 구간: {len(target_time_gaps)}개")
    if target_time_gaps:
        previous_time, current_time, gap = target_time_gaps[0]
        print(
            f"첫 비연속 구간: {previous_time:%Y-%m-%d %H:%M:%S} → "
            f"{current_time:%Y-%m-%d %H:%M:%S} ({gap})"
        )
    print(f"정답 기준: target_wind_speed_ms >= {THRESHOLD_MS} m/s → 1, 그 외 → 0")
    print_summary("학습 (정답시각 2023-01-01~2024-11-30)", train_rows)
    print_summary("평가 (정답시각 2024-12-01~2024-12-31)", evaluation_rows)

    if evaluation_rows:
        first_row = evaluation_rows[0]
        print(
            "첫 평가행: "
            f"prediction_time={first_row['prediction_time']}, "
            f"target_time={first_row['target_time']}, "
            f"target_wind_speed_ms={first_row['target_wind_speed_ms']} m/s, "
            f"label={first_row['_label']}"
        )
    else:
        print("첫 평가행: 없음")

    output_columns = [
        "prediction_time",
        "target_time",
        "actual_wind_speed_ms",
        "actual_label",
        "current_wind_speed_ms",
        "always_0_prediction",
        "current_wind_threshold_prediction",
        "strong_wind_probability",
        "model_prediction",
        "model_prediction_0_20",
    ]
    with PREDICTIONS_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()
        for row_index, (row, probability) in enumerate(
            zip(evaluation_rows, strong_wind_probabilities)
        ):
            current_wind_speed = float(row["wind_speed_ms"])
            writer.writerow(
                {
                    "prediction_time": row["prediction_time"],
                    "target_time": row["target_time"],
                    "actual_wind_speed_ms": row["target_wind_speed_ms"],
                    "actual_label": row["_label"],
                    "current_wind_speed_ms": row["wind_speed_ms"],
                    "always_0_prediction": method_predictions["항상 0"][row_index],
                    "current_wind_threshold_prediction": method_predictions[
                        "현재 풍속 >= 4.0 m/s"
                    ][row_index],
                    "strong_wind_probability": f"{probability:.6f}",
                    "model_prediction": method_predictions[
                        "로지스틱 회귀 (확률 >= 0.50)"
                    ][row_index],
                    "model_prediction_0_20": method_predictions[
                        "로지스틱 회귀 (확률 >= 0.20)"
                    ][row_index],
                }
            )

    print(f"예측 저장: {PREDICTIONS_PATH.name} ({len(evaluation_rows)}행)")
    score_columns = [
        "method",
        "evaluation_rows",
        "actual_strong_wind_count",
        "hits",
        "misses",
        "false_alarms",
        "correct_rejections",
        "accuracy",
        "strong_wind_precision",
        "strong_wind_recall",
    ]
    with SCORES_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=score_columns)
        writer.writeheader()
        for method, score in scores.items():
            writer.writerow(
                {
                    "method": method,
                    **{
                        key: (
                            f"{value:.6f}"
                            if isinstance(value, float)
                            else value
                        )
                        for key, value in score.items()
                    },
                }
            )

    evaluation_count = next(iter(scores.values()))["evaluation_rows"]
    actual_strong_wind_count = next(iter(scores.values()))[
        "actual_strong_wind_count"
    ]
    print(
        f"동일 평가행 확인: 세 방법 모두 {evaluation_count}행, "
        f"실제 강풍 {actual_strong_wind_count}행"
    )
    print(f"점수 저장: {SCORES_PATH.name}")
    comparison_methods = (
        "로지스틱 회귀 (확률 >= 0.50)",
        "로지스틱 회귀 (확률 >= 0.20)",
    )
    print(f"{'지표':<12} {'확률 >= 0.50':>14} {'확률 >= 0.20':>14}")
    for metric, format_value in (
        ("적중", lambda score: str(score["hits"])),
        ("놓침", lambda score: str(score["misses"])),
        ("오경보", lambda score: str(score["false_alarms"])),
        (
            "정밀도",
            lambda score: (
                "해당없음"
                if isinstance(score["strong_wind_precision"], str)
                else f"{score['strong_wind_precision'] * 100:.2f}%"
            ),
        ),
        (
            "재현율",
            lambda score: (
                "해당없음"
                if isinstance(score["strong_wind_recall"], str)
                else f"{score['strong_wind_recall'] * 100:.2f}%"
            ),
        ),
    ):
        print(
            f"{metric:<12} "
            f"{format_value(scores[comparison_methods[0]]):>14} "
            f"{format_value(scores[comparison_methods[1]]):>14}"
        )


if __name__ == "__main__":
    main()
