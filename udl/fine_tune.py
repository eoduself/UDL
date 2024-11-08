from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses
import os

def fine_tune(retrieval_model_name, dataset, document, gen_queries, gen_qrels, 
              batch_size=8, num_epochs = 1, evaluation_steps = 5000, warm_up_rate=0.1):
    
    print("Retrieval model for fine-tuning:",retrieval_model_name)
    model = SentenceTransformer(retrieval_model_name)
    retriever = TrainRetriever(model=model, batch_size=batch_size)
    train_samples = retriever.load_train(document, gen_queries, gen_qrels)

    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join("output", "{}-v1-{}".format(retrieval_model_name, dataset))
    os.makedirs(model_save_path, exist_ok=True)

    #### Hyperparameters for fine-tuning the retrieval model    
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * warm_up_rate)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=ir_evaluator,
                    epochs=num_epochs,
                    output_path=model_save_path,
                    warmup_steps=warmup_steps,
                    evaluation_steps=evaluation_steps,
                    use_amp=True)
    
    return model_save_path
    