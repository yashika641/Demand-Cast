import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export default function Drawer({ isOpen, onClose, title, children, position = 'right' }) {
  const positionClasses = {
    right: 'right-0 inset-y-0',
    left: 'left-0 inset-y-0',
    bottom: 'bottom-0 inset-x-0'
  };

  const slideVariants = {
    right: { x: '100%' },
    left: { x: '-100%' },
    bottom: { y: '100%' }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black bg-opacity-50"
            onClick={onClose}
          />
          <motion.div
            initial={slideVariants[position]}
            animate={{ x: 0, y: 0 }}
            exit={slideVariants[position]}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={`absolute ${positionClasses[position]} bg-white shadow-2xl ${
              position === 'bottom' ? 'max-h-[80vh] rounded-t-2xl' : 'w-full max-w-md'
            } flex flex-col`}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <h2 className="text-gray-900">{title}</h2>
              <button
                onClick={onClose}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-600" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-6 py-6">
              {children}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
