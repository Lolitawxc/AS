require 'image'

-- dataset squares :
-- 250 images toutes noires de taille 4 x 4 (indices 1 a 250)
-- 250 images toutes blanches de taille 4 x 4 (indices 251 a 500)
-- 1 image toute blanche sauf un pixel en bas a droite noir (indice 501, l'image à générer/compléter par pixelRNN, elle devrait alors devenir toute blanche)

squares = torch.Tensor(501, 1, 4, 4)
for k = 250,501 do 
    for i = 1,4 do
        for j = 1,4 do
            squares[k][1][i][j] = 255
        end
    end
end
squares[501][1][4][4] = 0
-- image.save('test.jpg', squares[501])
