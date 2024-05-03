import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random
import scala.io.Source
import java.io._
import scala.collection.mutable.{Set, ArrayBuffer}
import java.math.BigInteger


object task1 {
  val bitLen = 69997
  val hNum = 30
  val random = new Random()
  val hashAB = Array.fill(hNum)((random.nextInt(bitLen) + 1, random.nextInt(bitLen + 1)))
  // Define the hash function
  def myHashes(s: String): Array[Int] = {
    val p: BigInt = BigInt(100003)// Close approximation to 1e5 + 3
    val userInt = BigInt(s.getBytes("UTF-8").map("%02x".format(_)).mkString, 16)
    hashAB.map { case (a, b) => ((BigInt(a) * userInt + b) % p % bitLen).toInt }
  }


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Task1")//.setMaster("local[*]")
    val sc = new SparkContext(conf)
    val inputs = args(0)
    val streamSize = args(1).toInt
    val numAsks = args(2).toInt
    val outputs = args(3)

    val bx = new Blackbox

    val start_time = System.currentTimeMillis()
    sc.setLogLevel("ERROR")

    val bitArray = Array.fill(bitLen)(false)
    var total = 0
    val pUser = Set.empty[String] // Previously seen users
    val resultHeader = new StringBuilder("Time,FPR\n")

    for (i <- 1 to numAsks) {
      var fp = 0

      val streamUsers = bx.ask(inputs, streamSize)
      val userS = sc.parallelize(streamUsers)
      val userH = userS.map(user => (user, myHashes(user))).collect()

      userH.foreach { case (user, hashes) =>
        val isNew = hashes.exists(hash => !bitArray(hash))
        if (isNew) hashes.foreach(hash => bitArray(hash) = true)
        else if (!pUser.contains(user)) fp += 1
        pUser.add(user)
        total += 1
      }

      val fpr = fp.toDouble / streamSize
      resultHeader.append(s"$i,$fpr\n")
    }

    val end_time = System.currentTimeMillis()
    val execution = (end_time - start_time) / 1000
    println(s"Duration: $execution")

    new PrintWriter(outputs) { write(resultHeader.toString); close() }
    sc.stop()
  }
}
