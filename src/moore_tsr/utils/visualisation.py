import plotly.graph_objects as go
import pandas as pd

def plot_training_metrics(losses: list, save_path: str = None):
    """
    Trace la courbe de perte pendant l'entraînement avec Plotly.

    Args:
        losses (list): Liste des valeurs de perte.
        save_path (str, optional): Chemin pour sauvegarder le graphique.
    """
    # Création d'un DataFrame pour faciliter les calculs
    df = pd.DataFrame(losses, columns=["loss"])
    
    # Calcul de la moyenne mobile
    window_size = max(1, len(losses) // 20)
    df["smooth_loss"] = df["loss"].rolling(window=window_size, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Tracé de la perte brute
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["loss"],
            mode="lines",
            name="Perte d'entraînement",
            line=dict(color="blue", width=1),
        )
    )
    
    # Tracé de la moyenne mobile
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["smooth_loss"],
            mode="lines",
            name=f"Moyenne mobile (fenêtre={window_size})",
            line=dict(color="red", width=2),
        )
    )
    
    fig.update_layout(
        title="Évolution de la perte pendant l'entraînement",
        xaxis_title="Étapes",
        yaxis_title="Perte",
        legend_title="Légende",
        template="plotly_white",
    )
    
    if save_path:
        fig.write_image(save_path)
    fig.show()