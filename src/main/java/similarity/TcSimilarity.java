package similarity;

import net.librec.conf.Configuration;
import net.librec.conf.Configured;
import net.librec.data.DataModel;
import net.librec.data.convertor.TextDataConvertor;
import net.librec.math.structure.SparseMatrix;
import net.librec.similarity.AbstractRecommenderSimilarity;
import test.MainTest;

import java.util.List;

public class TcSimilarity extends AbstractRecommenderSimilarity {


    private Configuration conf;

    @Override
    protected double getSimilarity(List<? extends Number> thisList, List<? extends Number> thatList) {
        return 0;
    }

    public void buildSimilarityMatrix(Configuration conf, DataModel dataModel) throws Exception {
        String dirName = "ml-100k";
        conf.set(Configured.CONF_DATA_COLUMN_FORMAT, "UIRT");
        conf.set("inputDataPath", conf.get("dfs.data.dir") + "/" + dirName +  "/u1.base");

        TextDataConvertor textDataConverter = new TextDataConvertor(conf.get("inputDataPath"));
        textDataConverter.processData();

        SparseMatrix preferenceMatrix = textDataConverter.getPreferenceMatrix();
        SparseMatrix datetimeMatrix = textDataConverter.getDatetimeMatrix();

        System.out.println(datetimeMatrix.get(0,0));



    }
}
