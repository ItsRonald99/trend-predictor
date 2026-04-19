.PHONY: features train tune backtest thresholds

SYMS=QQQ VFV.TO XEQT.TO

features:
	@echo "🔧 Building features..."
	tp features --symbols $(SYMS)

train:
	@echo "🤖 Training baselines..."
	tp train --symbols $(SYMS)

tune:
	@echo "🔬 Tuning hyperparameters..."
	tp tune --symbols $(SYMS)

backtest:
	@echo "📈 Running backtests..."
	tp backtest --symbols QQQ --kind ridge_reg
	tp backtest --symbols QQQ --kind logit_cls --tuned

thresholds:
	@echo "🎯 Calibrating thresholds..."
	tp thresholds --symbols QQQ --kind logit_cls --cal-frac 0.8
