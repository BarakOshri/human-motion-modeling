THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 128 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 0.8 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden128lr0.8tanh --print_console 1
THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 128 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 0.8 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden128lr0.8relu --print_console 1 --activation relu
THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 256 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 0.8 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden256lr0.8tanh --print_console 1
THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 256 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 0.8 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden256lr0.8relu --print_console 1 --activation relu
THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 512 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 0.8 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden512lr0.8tanh --print_console 1 --activation relu
THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 512 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 1.4 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden512lr1.4relu --print_console 1
THEANO_FLAGS='device=gpu0' python train_lstm_mit.py --inputs frame --output frame --n_hidden 512 --epochs 7000 --joi 0 5 9 --batch_size 400 --learning_rate 1.4 --percent_train 0.8 --num_samples all --final_frame_lookahead 100 --load_data 0 --load_model 0 --output_subdirectory predframehidden512lr1.4reluadadelta --print_console 1 --optimize_method adadelta
