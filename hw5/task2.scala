import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random
import scala.io.Source
import java.io._
import scala.collection.mutable.{Set, ArrayBuffer}
import java.math.BigInteger

object task2 {
  val bitLen = 2345
  val hNum = 30
  val random = new Random() // Ensure reproducibility by setting a seed
  val hashAB: Array[(Int, Int)] = Array.fill(hNum)((random.nextInt(bitLen) + 1, random.nextInt(bitLen + 1)))

  def myHashes(s: String): Array[Int] = {
    val p: BigInt = BigInt(100003)// Close approximation to 1e5 + 3
    val userInt = BigInt(s.getBytes("UTF-8").map("%02x".format(_)).mkString, 16)
    hashAB.map { case (a, b) => ((BigInt(a) * userInt + b) % p % bitLen).toInt }
  }

  def countTrails(b: BigInt): Int = {
    b.toString(2).reverse.takeWhile(_ == '0').length
  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Task2")
    val sc = new SparkContext(conf)

    val start_time = System.currentTimeMillis()
    sc.setLogLevel("ERROR")

    val inputs = args(0)
    val streamSize = args(1).toInt
    val numAsks = args(2).toInt
    val outputs = args(3)

    val bx = new Blackbox()

    var result = new ArrayBuffer[(Int, Int, Double)]()
    var ground: Int = 0
    var estSum: Double = 0.0
    for (i <- 1 to numAsks) {
      val streamUsers = bx.ask(inputs, streamSize)
      val userHashesRDD = sc.parallelize(streamUsers).map(user => myHashes(user))

      val maxTrails = userHashesRDD.map(_.map(hash => countTrails(BigInt(hash))))
        .reduce((a, b) => a.zip(b).map { case (x, y) => Math.max(x, y) })

      val estimate = ((maxTrails.map(z => Math.pow(2, z)).sum / maxTrails.length).floor).toInt
      ground += streamUsers.distinct.length
      estSum += estimate
      result += ((i, streamUsers.distinct.length, estimate))
    }
    //println(estSum / ground)
    val writer = new PrintWriter(new File(outputs))

    writer.println("Time,Ground Truth,Estimation")
    result.foreach { case (time, groundTruth, estimate) =>
      writer.println(f"$time,$groundTruth,${estimate.toInt}%d")
    }

    writer.close()
    val end_time = System.currentTimeMillis()
    println(s"Duration: ${(end_time - start_time) / 1000}")

    sc.stop()
  }

}
