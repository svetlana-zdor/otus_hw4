import json
import chromadb
import ollama

# Генеративная модель Ollama
LLM_MODEL = "qwen2.5"
COLLECTION_NAME = "space_history"

def setup_chromadb(dataset_path: str = "dataset.json"):
    """Инициализирует ChromaDB и автоматически векторизует документы."""
    client = chromadb.Client()
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
        
    collection = client.create_collection(name=COLLECTION_NAME)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Индексация документов в ChromaDB (векторы создаются автоматически)...")
    
    # ChromaDB самостоятельно создает эмбеддинги для переданных документов
    collection.add(
        ids=[str(item["id"]) for item in dataset],
        documents=[item["text"] for item in dataset],
        metadatas=[{"topic": item["topic"]} for item in dataset]
    )
    
    print(f"Успешно заиндексировано документов: {len(dataset)}\n")
    return collection

def retrieve_context(query: str, collection, top_k: int = 2) -> list[str]:
    """Поиск релевантных документов через векторный поиск ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(query: str, context_docs: list[str]) -> str:
    """Генерация ответа моделью Qwen 2.5 на основе контекста."""
    context_str = "\n\n".join(context_docs)
    
    prompt = (
        f"Ты — полезный ассистент. Ответь на вопрос пользователя, используя ТОЛЬКО предоставленный контекст.\n"
        f"Если в контексте нет ответа, ответь: 'К сожалению, в базе знаний нет информации по данному вопросу.'\n\n"
        f"КОНТЕКСТ:\n{context_str}\n\n"
        f"ВОПРОС: {query}\n\n"
        f"ОТВЕТ:"
    )
    
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt
    )
    
    return response["response"].strip()

def main():
    collection = setup_chromadb("dataset.json")
    
    print("RAG-система готова к работе! Введите 'выход' для завершения.\n")
    
    while True:
        query = input("Задайте вопрос по базе знаний: ").strip()
        if not query or query.lower() in ["exit", "quit", "выход"]:
            break
            
        print("\n[1/2] Поиск релевантного контекста в базе...")
        context_docs = retrieve_context(query, collection, top_k=2)
        
        print("\n--- Найденный контекст ---")
        for i, doc in enumerate(context_docs, 1):
            print(f"Документ #{i}:\n{doc}\n")
            
        print("[2/2] Генерация ответа через Qwen 2.5...")
        answer = generate_answer(query, context_docs)
        
        print("\n=== ОТВЕТ RAG ===")
        print(answer)
        print("=" * 40 + "\n")

if __name__ == "__main__":
    main()