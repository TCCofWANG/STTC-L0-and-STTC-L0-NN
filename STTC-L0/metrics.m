function [rmae,rmse,mae] = metrics(X,X_approx,idx_recover)
rmae = sumabs(double((X-X_approx).*idx_recover))/sumabs(double(X.*idx_recover));
        
squaredDiff = ((X-X_approx).*idx_recover) .^ 2;
meanSquaredDiff = sum(double(squaredDiff),'all')/sum(double(idx_recover),'all');
rmse = sqrt(meanSquaredDiff);


absDiff = abs(double((X-X_approx).*idx_recover));
mae = sum(double(absDiff),'all')/sum(double(idx_recover),'all');
end

