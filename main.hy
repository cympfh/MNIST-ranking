(import
  [tensorflow :as tf]
  [numpy]
  [keras.backend [tensorflow_backend :as K]]
  [keras.datasets [mnist]]
  [keras.layers [Activation Dense Flatten Reshape Input Lambda]]
  [keras.layers.convolutional [Conv2D]]
  [keras.layers.merge [add]]
  [keras.models [Sequential Model]])


;; data
(def [[x_train y_train] [x_test y_test]] (mnist.load_data))
(setv x_train (/ (.astype x_train "f") 255))
(setv x_test (/ (.astype x_test "f") 255))

(defn data-generator [x y]
      (def n (len x))
      (for [i (range 10000000000)]
           (def idx1 (% i n)
                idx2 (% (+ i (// n 2)) n)
                x1 (get x idx1)
                x2 (get x idx2)
                y1 (get y idx1)
                y2 (get y idx2)
                pr (if (< y1 y2) 1 (if (> y1 y2) 0 0.5)))
           (yield [[x1 x2] pr])))

(defn batch-generator [x y &optional [batch-size 30]]
      (def X1 (numpy.zeros (, batch-size 28 28))
           X2 (numpy.zeros (, batch-size 28 28))
           Y (numpy.zeros (, batch-size 1)))
      (def i 0)
      (for [[[x1 x2] pr] (data-generator x y)]
           (setv (get X1 i) x1
                 (get X2 i) x2
                 (get Y i) pr)
           (setv i (inc i))
           (if (= i batch-size)
             (do
               (setv i 0)
               (yield [[X1 X2] Y])))))


;; model
(def score-predictor
     (doto
       (Sequential)
       (.add (Reshape (, 28 28 1) :input_shape (, 28 28)))
       (.add (Conv2D 8 (, 5 5) :strides (, 2 2) :activation 'relu))
       (.add (Conv2D 16 (, 5 5) :strides (, 2 2) :activation 'relu))
       (.add (Flatten))
       (.add (Dense 10 :activation 'relu))
       (.add (Dense 1))
       (.summary)))


(def model
     (do

       (defn loss [pr o]
             (- (K.log (+ (K.exp o) 1))
                (* pr o)))

       (defn mse [pr o]
             (K.mean
               (K.square (- pr (K.sigmoid o)))
               :axis -1))

       (def subtract (Lambda (fn [[x y]] (- x y))))
       (def x1 (Input (, 28 28))
            x2 (Input (, 28 28))
            f1 (score-predictor x1)
            f2 (score-predictor x2)
            o (subtract [f1 f2]))
       (doto
         (Model :inputs [x1 x2] :outputs o)
         (.compile :optimizer 'adam :loss loss :metrics [mse])
         (.summary))))


;; learning
(do
  (def batch-size 30)
  (.fit_generator model
                  (batch-generator x_train y_train :batch-size 30)
                  :steps-per-epoch (// 60000 batch-size)
                  :epochs 5
                  :verbose 1
                  :validation-data (batch-generator x_test y_test :batch-size 30)
                  :validation-steps (// 10000 batch-size)))

;; testing
(import
  [PIL [Image ImageDraw]])

(defn float->str [f]
      (str (/ (int (* f 100)) 100)))

(defn array->img [arr]
      (Image.fromarray (.astype (* arr 255) numpy.uint8)))

(do
  (def batch-size 20
       margin 6)
  (for [[[X _] _] (batch-generator x_test y_test :batch-size batch-size)]
       (def F (.predict score-predictor X))
       (def d (list (map (fn [i] [(float (get F i 0)) (get X i)]) (range batch-size))))
       (.sort d)

       (def canvas (Image.new "RGBA" (, (+ margin (* (+ 28 margin) 20)) (+ 28 (* 3 margin) 12)) (, 255 255 255))
            draw (ImageDraw.Draw canvas))

       (for [[i [f x]] (enumerate d)]
            (def x-pos (+ margin (* i (+ 28 margin))))
            (.paste canvas (array->img x) (, x-pos margin))
            (draw.text (, x-pos (+ 28 margin margin)) (float->str f) (, 0 0 0)))

       (.save canvas "out.png")
       (break)))
