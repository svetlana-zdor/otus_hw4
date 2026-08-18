import json
import ollama

# Указываем модель Qwen 2.5
MODEL_NAME = "qwen2.5"

# 10 подтем для генерации базы знаний (История освоения космоса)
TOPICS = [
    "Запуск первого искусственного спутника Земли (Спутник-1)",
    "Первый полёт человека в космос (Юрий Гагарин)",
    "Программа «Аполлон» и высадка человека на Луну",
    "Орбитальная станция «Мир»",
    "Международная космическая станция (МКС)",
    "Марсоходы серии Mars Exploration Rover и Curiosity",
    "Космический телескоп «Хаббл»",
    "Космический телескоп «Джеймс Уэбб»",
    "Развитие частной космонавтики и компания SpaceX",
    "Перспективы пилотируемой миссии на Марс"
]

def generate_note(topic: str, index: int) -> dict:
    """Генерирует одну заметку с помощью Qwen 2.5."""
    print(f"[{index}/10] Генерация заметки: '{topic}'...")
    
    prompt = (
        f"Напиши информативную заметку на тему '{topic}'. "
        f"Длина текста должна составлять около 100 слов. "
        f"Пиши только факты на русском языке. Не добавляй вступительных и заключительных фраз."
    )
    
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt
    )
    
    text = response['response'].strip()
    
    return {
        "id": index,
        "topic": topic,
        "text": text
    }

def main():
    dataset = []
    
    print(f"Запуск генерации датасета с помощью модели {MODEL_NAME}...\n")
    for i, topic in enumerate(TOPICS, start=1):
        try:
            note = generate_note(topic, i)
            dataset.append(note)
        except Exception as e:
            print(f"Ошибка при генерации темы '{topic}': {e}")
            
    output_filename = "dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\nГотово! Сгенерировано {len(dataset)} заметок и сохранено в '{output_filename}'.")

if __name__ == "__main__":
    main()