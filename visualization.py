import matplotlib.pyplot as plt

def plot_results(results):
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    train_times = [results[model]['train_time'] for model in models]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(models, accuracies)
    plt.title('Accuracy')
    
    plt.subplot(1, 3, 2)
    plt.bar(models, f1_scores)
    plt.title('F1-Score')
    
    plt.subplot(1, 3, 3)
    plt.bar(models, train_times)
    plt.title('Training Time')
    
    plt.show()

plot_results(results)
