%% Drowsiness ANFIS Training Script with Data Augmentation
clc; clear; close all;

%% --- Load Dataset ---
data = readmatrix('dataset.csv');  % Columns: Timestamp, EAR, MAR, PERCLOS, HeadPitch, Drowsy_Label
features = data(:, 2:5);           % EAR, MAR, PERCLOS, HeadPitch
labels = data(:, 6);

%% --- Normalize Features to [0, 1] ---
minF = min(features);
maxF = max(features);
features_norm = (features - minF) ./ (maxF - minF + eps);  % Avoid division by zero
dataset = [features_norm labels];

%% --- Save normalization constants for Python ---
save('feature_norm_constants.mat', 'minF', 'maxF');

%% --- 3-Way Split: 60% Train, 20% Check, 20% Test ---
n = size(dataset, 1);
idx = randperm(n);

train_end = round(0.6 * n);
check_end = round(0.8 * n);

train_data = dataset(idx(1:train_end), :);
check_data = dataset(idx(train_end+1:check_end), :);
test_data  = dataset(idx(check_end+1:end), :);

disp(['Training samples: ', num2str(size(train_data,1))]);
disp(['Checking samples: ', num2str(size(check_data,1))]);
disp(['Testing samples: ', num2str(size(test_data,1))]);

%% --- Data Augmentation for Training Set ---
disp('Applying data augmentation...');

aug_factor = 5;      % How many times to augment training data
noise_level = 0.02;  % ~2% Gaussian noise

aug_train_features = [];
aug_train_labels   = [];
n_train = size(train_data,1);

for k = 1:aug_factor
    for i = 1:n_train
        sample = train_data(i,1:4);
        label  = train_data(i,5);

        % Add small Gaussian noise
        noisy_sample = sample .* (1 + noise_level*randn(1,4));
        noisy_sample = max(min(noisy_sample,1),0);  % Clip to [0,1]

        % Optional interpolation with a random training sample
        j = randi(n_train);
        interp_sample = 0.7*noisy_sample + 0.3*train_data(j,1:4);
        interp_sample = max(min(interp_sample,1),0);

        % Collect augmented samples
        aug_train_features = [aug_train_features; interp_sample];
        aug_train_labels   = [aug_train_labels; label];
    end
end

% Combine original + augmented training data
train_features_aug = [train_data(:,1:4); aug_train_features];
train_labels_aug   = [train_data(:,5); aug_train_labels];
train_data_aug = [train_features_aug train_labels_aug];

disp(['Original training samples: ', num2str(n_train)]);
disp(['Augmented training samples: ', num2str(size(train_data_aug,1))]);

%% --- Generate Initial FIS ---
numMFs = 3;  % Membership functions per input
fismat = genfis1(train_data_aug, numMFs, 'gbellmf');

%% --- Train ANFIS ---
numEpochs = 30;
[fis, trainError, stepSize, chkFIS, chkError] = ...
    anfis(train_data_aug, fismat, [numEpochs 0 0.01 0.9 1.1], [], check_data);

%% --- Save Trained FIS ---
writefis(fis, 'drowsy_fis_model_augmented');
disp('Trained FIS saved as "drowsy_fis_model_augmented.fis"');

%% --- Convergence Plot ---
figure;
plot(1:numEpochs, trainError, 'b-o','LineWidth',1.5); hold on;
plot(1:numEpochs, chkError, 'r-s','LineWidth',1.5);
xlabel('Epochs'); ylabel('Error (RMSE)');
title('ANFIS Convergence Plot');
legend('Training Error','Checking Error');
grid on;

%% --- Validity / Performance Plot on Test Data ---
test_inputs = test_data(:,1:4);
test_targets = test_data(:,5);
test_outputs = evalfis(fis, test_inputs);

figure;
scatter(test_targets, test_outputs, 50,'filled'); hold on;
plot([0 1],[0 1],'k--','LineWidth',1.5);
xlabel('Actual Drowsiness Label');
ylabel('Predicted Drowsiness Score');
title('ANFIS Prediction vs Actual (Validity)');
grid on; axis([0 1 0 1]);

%% --- Display ANFIS Structure ---
disp('--- ANFIS Structure ---');
disp(['Number of Inputs: ', num2str(numel(fis.input))]);
disp(['Number of Outputs: ', num2str(numel(fis.output))]);
disp(['Number of Rules: ', num2str(numel(fis.rule))]);

for i = 1:numel(fis.input)
    fprintf('\nInput %d: %s\n', i, fis.input(i).name);
    for j = 1:numel(fis.input(i).mf)
        fprintf('  MF %d: %s (%s)\n', j, fis.input(i).mf(j).name, fis.input(i).mf(j).type);
    end
end

for i = 1:numel(fis.output)
    fprintf('\nOutput %d: %s\n', i, fis.output(i).name);
    for j = 1:numel(fis.output(i).mf)
        fprintf('  MF %d: %s (%s)\n', j, fis.output(i).mf(j).name, fis.output(i).mf(j).type);
    end
end
%% --- Plot Input Membership Functions ---
figure;
subplot(2,2,1); plotmf(fis,'input',1); title('EAR MFs');
subplot(2,2,2); plotmf(fis,'input',2); title('MAR MFs');
subplot(2,2,3); plotmf(fis,'input',3); title('PERCLOS MFs');
subplot(2,2,4); plotmf(fis,'input',4); title('Head Pitch MFs');
