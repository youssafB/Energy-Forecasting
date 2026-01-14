
from src.pipelines.pipeline import run_pipeline
from src.models.save_model import save_model
from src.evaluation.plots import plot_predictions





def run_experiments(configs, y_true_col=None):
    best_score = float('inf')
    best_model = None
    best_preds = None

    for cfg in configs:
        print(f"\nRunning experiment: {cfg['name']}")
        ml, score, preds = run_pipeline(cfg)

        print(f"Score for {cfg['name']}: {score}")

        if score < best_score:
            best_score = score
            best_model = ml
            best_preds = preds
            save_model(best_model, f"models/best_model_{cfg['name']}.joblib")
            print(f"✅ New best model saved: {cfg['name']} with score {best_score}")

            if y_true_col:
                plot_predictions(cfg[y_true_col], best_preds, title=f"Best Model: {cfg['name']}")

    print(f"\n✅ Best score across all experiments: {best_score}")
    return best_model, best_preds
