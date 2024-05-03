import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.graphx._
import org.apache.spark.sql.functions._
import org.graphframes.GraphFrame
import scala.io.Source
import java.io.{PrintWriter, File}

object task1 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Task1")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder
      .appName("community")
      .getOrCreate()
    sc.setLogLevel("ERROR")
    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._
    val thr = args(0).toInt
    val inputs = args(1)
    val output = args(2)


    val start_time = System.currentTimeMillis()

    // Load and preprocess the data
    val csv_rdd = sc.textFile(inputs)
    val header = csv_rdd.first()
    val data_nh = csv_rdd.filter(_ != header).map(_.split(","))
      .map(attributes => (attributes(0), Set(attributes(1))))
      .reduceByKey(_ union _)

    val data_pre = data_nh.filter(_._2.size >= thr).collectAsMap()
    val uni_user = data_pre.keys.toList
    val combuser = uni_user.combinations(2).toList

    val filter_p = sc.parallelize(combuser).filter(users => data_pre(users(0)).intersect(data_pre(users(1))).size >= thr)
    val edges_n = filter_p.flatMap(users => Seq((users(1), users(0)), (users(0), users(1))))
    val nodes = filter_p.flatMap(users => Seq(users(0), users(1))).distinct().map(user => (user))

    // Create vertices and edges DataFrames
    val vertices = nodes.toDF("id")
    val edges = edges_n.toDF("src", "dst")
    val g = GraphFrame(vertices, edges)

    // Apply the Label Propagation algorithm
    val result = g.labelPropagation.maxIter(5).run()
    val finalResult = result.rdd
      .map(row => (row.getAs[Long]("label"), Array(row.getAs[String]("id"))))
      .reduceByKey(_ ++ _)
      .map { case (label, ids) => (ids.sorted, ids.length) }
      .sortBy { case (ids, size) => (size, ids(0)) }

    val fin = finalResult.map(_._1)

    // Write the communities to a text file
    val writer = new PrintWriter(new File(output))
    fin.collect().foreach(f => {
      val com = f.map(u => s"'$u'").mkString(", ")
      writer.write(com + "\n")
    })
    writer.close()

    // Stop SparkSession
    spark.stop()
    sc.stop()

    val end_time = System.currentTimeMillis()
    val execution = (end_time - start_time) / 1000.0
    println(s"Duration: $execution")

  }
}
