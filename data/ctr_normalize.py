class CTRNormalize:
    @staticmethod
    def cutoff(clicks: float, count: float, eps: float):
        count = max(count, eps)
        clicks = min(max(clicks, 0), count)
        return clicks / count

    @staticmethod
    def no_action(clicks: float, count: float, eps: float):
        return clicks / count

    # @staticmethod
    # def smoothing(clicks: float, count: float):
    #     ctr = clicks / count
    #     return (ctr * count + prior_weight * prior) / (count + prior_weight)
