package common;

import cluster.KMeansClustering;
import net.librec.conf.Configuration;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.similarity.AbstractRecommenderSimilarity;
import recommender.KMeansKNNRecommender;


public class TestUtils {

    public static void testMAEWithKmeans(Configuration conf, RecommenderContext context, String knn) throws Exception {
        conf.set("rec.neighbors.knn.number", knn);

        KMeansClustering.invokeClustering(context.getDataModel(), (AbstractRecommenderSimilarity) context.getSimilarity(), 50, 10);

        KMeansKNNRecommender recommender = new KMeansKNNRecommender();

        recommender.setContext(context);
        recommender.recommend(context);

        RecommenderEvaluator evaluator = new MAEEvaluator();
        System.out.println(knn + "MAE" + recommender.evaluate(evaluator));
    }
}
