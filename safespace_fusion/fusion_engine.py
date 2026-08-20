class FusionEngine:
    def __init__(self, lambda_=0.3, alpha=0.7, beta=0.3):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.modalities = ['biomedical', 'behavioral', 'text', 'voice', 'facial']

        # Initialize EMA, residual, and persistence for each modality
        self.ema = {mod: None for mod in self.modalities}
        self.residual = {mod: None for mod in self.modalities}
        self.persistence = {mod: None for mod in self.modalities}

    def run_batch(self, batch_scores):
        results = []

        for idx, scores in enumerate(batch_scores):
            # First-time setup for EMA
            if idx == 0 and any(self.ema[mod] is None for mod in self.modalities):
                self.ema = {mod: scores[mod] for mod in self.modalities}

            weights = {}
            updated_ema = {}

            for mod in self.modalities:
                S = scores[mod]
                EMA_prev = self.ema[mod]
                EMA_new = self.lambda_ * S + (1 - self.lambda_) * EMA_prev

                residual = S - EMA_new
                persistence = EMA_new

                # Weighted score computation
                weight = max(0, self.alpha * residual + self.beta * persistence)
                weights[mod] = weight
                updated_ema[mod] = EMA_new

                # ✅ Save internal details
                self.residual[mod] = residual
                self.persistence[mod] = persistence

            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {mod: weights[mod] / total_weight for mod in self.modalities}

            # Final WMSS score
            WMSS = sum(scores[mod] * normalized_weights[mod] for mod in self.modalities)

            # Label classification
            if WMSS < 0.4:
                stress_level = "Low"
            elif WMSS <= 0.7:
                stress_level = "Medium"
            else:
                stress_level = "High"

            # Most influential modality
            dominant_modality = max(normalized_weights, key=normalized_weights.get)

            # Store result
            results.append({
                'WMSS': round(WMSS, 4),
                'Stress Level': stress_level,
                'Dominant Modality': dominant_modality,
                'Normalized Weights': {k: round(v, 4) for k, v in normalized_weights.items()}
            })

            # Update EMA for next timestep
            self.ema = updated_ema

        return results
